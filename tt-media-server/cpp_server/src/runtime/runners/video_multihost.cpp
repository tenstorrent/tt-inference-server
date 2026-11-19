// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/video_multihost.hpp"

#include <pybind11/embed.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>

#include "runtime/dit/dit_python_runner.hpp"
#include "runtime/runners/video_python_env.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;

namespace tt::runners {

namespace {

// A follower spends its life blocked in an MPI receive inside ttnn. A signal
// handler that only sets a flag (Python's SIGINT handler) is never serviced
// there, so SIGTERM and SIGINT (mpirun forwards them) end the process at once,
// as the Python follower's install_exit_signal_handlers did.
void exitOnSignal(int sig) { ::_exit(128 + sig); }

void installExitSignalHandlers() {
  struct sigaction action{};
  action.sa_handler = exitOnSignal;
  sigemptyset(&action.sa_mask);
  ::sigaction(SIGTERM, &action, nullptr);
  ::sigaction(SIGINT, &action, nullptr);
}

int envInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  return std::atoi(value);
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
  // Without Python's own handlers, then ours: Python's SIGINT handler would
  // replace the exit handler.
  py::initialize_interpreter(false);
  installExitSignalHandlers();
  try {
    py::gil_scoped_acquire gil;
    prepareVideoPythonPath();
    // MODEL_RUNNER, DEVICE and MODEL_WEIGHTS_PATH come from the environment
    // tt-run gives every rank.
    const int code = tt::dit::runFollowerRank(rank);
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
