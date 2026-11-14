// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/video_multihost.hpp"

#include <pybind11/embed.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

#include "utils/logger.hpp"

namespace py = pybind11;

namespace tt::runners {

namespace {

int envInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  return std::atoi(value);
}

void prependEntry(py::list sysPath, const std::string& value) {
  if (value.empty()) return;
  for (const auto& entry : sysPath) {
    if (py::str(entry).cast<std::string>() == value) return;
  }
  sysPath.attr("insert")(0, py::str(value));
}

void prependSysPath(const char* envName) {
  const char* value = std::getenv(envName);
  if (!value || !*value) return;
  prependEntry(py::module_::import("sys").attr("path"), value);
}

void addMetalSitePackages() {
  const char* metalHome = std::getenv("TT_METAL_HOME");
  if (!metalHome || !*metalHome) return;
  namespace fs = std::filesystem;
  const fs::path libDir = fs::path(metalHome) / "python_env" / "lib";
  std::error_code ec;
  if (!fs::is_directory(libDir, ec)) return;
  py::module_ site = py::module_::import("site");
  for (const fs::directory_entry& child : fs::directory_iterator(libDir, ec)) {
    if (ec) return;
    const std::string name = child.path().filename().string();
    if (name.rfind("python", 0) != 0) continue;
    const fs::path sitePackages = child.path() / "site-packages";
    if (!fs::is_directory(sitePackages, ec)) continue;
    site.attr("addsitedir")(sitePackages.string());
    TT_LOG_INFO("[VideoMultihost] Added metal site-packages: {}",
                sitePackages.string());
  }
}

void preparePythonPath() {
  const char* pythonPath = std::getenv("TT_PYTHON_PATH");
  const char* metalHome = std::getenv("TT_METAL_HOME");
  if (!pythonPath || !*pythonPath) {
    throw std::runtime_error(
        "[VideoMultihost] TT_PYTHON_PATH is not set. Point it at the "
        "tt-media-server directory.");
  }
  if (!metalHome || !*metalHome) {
    throw std::runtime_error(
        "[VideoMultihost] TT_METAL_HOME is not set. Point it at the tt-metal "
        "checkout.");
  }
  prependSysPath("TT_METAL_HOME");
  prependSysPath("TT_PYTHON_PATH");
  addMetalSitePackages();
}

}  // namespace

std::optional<int> launchRank() {
  const char* value = std::getenv("OMPI_COMM_WORLD_RANK");
  if (!value || !*value) return std::nullopt;
  return std::atoi(value);
}

bool isMultihostLaunch() { return envInt("OMPI_COMM_WORLD_SIZE", 1) > 1; }

bool isFollowerRank() {
  const std::optional<int> rank = launchRank();
  return rank.has_value() && *rank != 0;
}

int runVideoFollowerRank() {
  const int rank = launchRank().value_or(-1);
  TT_LOG_INFO("[VideoMultihost] Follower rank {} opening its part of the mesh",
              rank);
  py::initialize_interpreter();
  try {
    py::gil_scoped_acquire gil;
    preparePythonPath();
    py::module_ followerMod = py::module_::import("multihost.follower");
    followerMod.attr("install_exit_signal_handlers")();
    py::object follower = followerMod.attr("MultiHostLockstepFollower")(rank);
    const int code = follower.attr("run")().cast<int>();
    TT_LOG_INFO("[VideoMultihost] Follower rank {} exited with {}", rank, code);
    return code;
  } catch (const py::error_already_set& e) {
    TT_LOG_ERROR("[VideoMultihost] Follower rank {} failed: {}", rank,
                 e.what());
    return 1;
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[VideoMultihost] Follower rank {} failed: {}", rank,
                 e.what());
    return 1;
  }
}

void startLauncherWatch() {
  if (!isMultihostLaunch() || isFollowerRank()) return;
  std::thread([] {
    const pid_t launcher = ::getppid();
    while (::getppid() == launcher) {
      ::sleep(2);
    }
    TT_LOG_ERROR(
        "[VideoMultihost] mpirun (pid {}) exited; shutting down rank 0",
        launcher);
    ::kill(::getpid(), SIGTERM);
  }).detach();
}

}  // namespace tt::runners
