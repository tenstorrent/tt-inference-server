// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/video_python_env.hpp"

#include <pybind11/embed.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

#include "utils/logger.hpp"

namespace py = pybind11;

namespace tt::runners {

namespace {

void prependEntry(py::list sysPath, const std::string& value) {
  if (value.empty()) return;
  for (const auto& entry : sysPath) {
    if (py::str(entry).cast<std::string>() == value) return;
  }
  sysPath.attr("insert")(0, py::str(value));
}

// The embedded interpreter does not pick up the shell PYTHONPATH, and a raw
// sys.path insert does not run .pth files. ttnn is an editable install: its
// .pth file maps the package onto the tt-metal checkout. site.addsitedir
// processes those files the same way the metal virtualenv does.
void addMetalSitePackages(const std::filesystem::path& metalHome) {
  namespace fs = std::filesystem;
  const fs::path libDir = metalHome / "python_env" / "lib";
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
    TT_LOG_INFO("[VideoPython] Added metal site-packages: {}",
                sitePackages.string());
  }
}

}  // namespace

std::filesystem::path resolveVideoPythonDir() {
  namespace fs = std::filesystem;
  if (const char* existing = std::getenv("TT_VIDEO_PYTHON_DIR")) {
    if (*existing) return existing;
  }
  std::error_code ec;
  const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  return ec ? fs::path("python") : exe.parent_path().parent_path() / "python";
}

void prepareVideoPythonPath() {
  namespace fs = std::filesystem;
  const char* metalHome = std::getenv("TT_METAL_HOME");
  if (!metalHome || !*metalHome) {
    throw std::runtime_error(
        "[VideoPython] TT_METAL_HOME is not set. Point it at the tt-metal "
        "checkout.");
  }
  const fs::path dir = resolveVideoPythonDir();
  std::error_code ec;
  if (!fs::is_regular_file(dir / "cpp_dit_adapters" / "__init__.py", ec)) {
    throw std::runtime_error(
        "[VideoPython] cpp_dit_adapters package not found in '" + dir.string() +
        "'. Set TT_VIDEO_PYTHON_DIR.");
  }
  py::list sysPath = py::module_::import("sys").attr("path");
  prependEntry(sysPath, metalHome);
  prependEntry(sysPath, dir.string());
  addMetalSitePackages(metalHome);
  TT_LOG_INFO("[VideoPython] Runner package directory: {}", dir.string());
}

}  // namespace tt::runners
