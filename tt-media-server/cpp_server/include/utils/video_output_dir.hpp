// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace tt::utils {

/**
 * Directory the video runner writes mp4 files to. `TT_VIDEO_OUTPUT_DIR` if set,
 * else `<binary dir>/../videos` (cpp_server/videos). The server process and the
 * worker resolve it the same way, so both see the same directory.
 */
inline std::filesystem::path resolveVideoOutputDir() {
  namespace fs = std::filesystem;
  if (const char* existing = std::getenv("TT_VIDEO_OUTPUT_DIR")) {
    if (*existing) return existing;
  }
  std::error_code ec;
  const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  return ec ? fs::path("videos") : exe.parent_path().parent_path() / "videos";
}

/**
 * Empty when files can be created in `dir`, else a short reason. Creates the
 * directory if it is missing, writes one byte to a probe file and removes it.
 * The runner writes the mp4 only after the whole generation, so a directory
 * that cannot be written to would otherwise cost a full run before the job
 * fails. Cheap enough to call for every request (a few syscalls).
 */
inline std::string checkDirWritable(const std::filesystem::path& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) return "cannot create '" + dir.string() + "': " + ec.message();
  static std::atomic<uint64_t> counter{0};
  const fs::path probe =
      dir / (".write-check-" + std::to_string(::getpid()) + "-" +
             std::to_string(counter.fetch_add(1)));
  errno = 0;
  {
    std::ofstream out(probe, std::ios::binary);
    if (!out) {
      return "cannot write in '" + dir.string() +
             "': " + std::strerror(errno != 0 ? errno : EACCES);
    }
    out << 'x';
    out.flush();
    if (!out) {
      return "cannot write in '" + dir.string() +
             "': " + std::strerror(errno != 0 ? errno : EIO);
    }
  }
  fs::remove(probe, ec);
  return {};
}

/**
 * True when `path` resolves (symlinks and `..` followed) to a file strictly
 * inside `root`. Used before deleting a path a runner reported.
 */
inline bool isInsideDir(const std::string& path,
                        const std::filesystem::path& root) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path p = fs::weakly_canonical(path, ec);
  if (ec) return false;
  const fs::path r = fs::weakly_canonical(root, ec);
  if (ec) return false;
  if (p == r) return false;
  auto pi = p.begin();
  for (auto ri = r.begin(); ri != r.end(); ++ri, ++pi) {
    if (pi == p.end() || *pi != *ri) return false;
  }
  return true;
}

/**
 * Deletes regular `*.mp4` files that sit directly in `root` (no recursion, no
 * symlinks followed). Meant for startup, when the in-memory job table is empty,
 * so every such file is an orphan of a previous run. Returns how many were
 * removed.
 */
inline size_t purgeOrphanMp4(const std::filesystem::path& root) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(root, ec)) return 0;
  size_t removed = 0;
  for (fs::directory_iterator it(root, ec), end; !ec && it != end;
       it.increment(ec)) {
    std::error_code e2;
    if (it->is_symlink(e2) || !it->is_regular_file(e2)) continue;
    if (it->path().extension() != ".mp4") continue;
    if (fs::remove(it->path(), e2)) ++removed;
  }
  return removed;
}

}  // namespace tt::utils
