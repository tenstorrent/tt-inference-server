// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace tt::utils {

/** Python `_FFMPEG_REMUX_TIMEOUT_S`. */
constexpr int kFaststartTimeoutSeconds = 60;

/**
 * Same as Python `VideoManager.ensure_faststart`:
 * `ffmpeg -y -i <in> -c copy -movflags faststart <out>`. No shell is used, the
 * paths are passed as argv. Returns true only when ffmpeg exits 0 within the
 * timeout and `output` is a non-empty file. On timeout ffmpeg is killed.
 */
inline bool ensureFaststart(const std::string& input, const std::string& output,
                            int timeoutSeconds = kFaststartTimeoutSeconds) {
  // posix_spawnp, not fork: the server has many threads (HTTP, Python, logger)
  // and a fork child that calls freopen/malloc can deadlock on a lock another
  // thread held at the time of the fork. The child's stdin/stdout/stderr point
  // at /dev/null (ffmpeg is chatty and must not read the server stdin).
  posix_spawn_file_actions_t actions;
  if (::posix_spawn_file_actions_init(&actions) != 0) return false;
  struct ActionsGuard {
    posix_spawn_file_actions_t* a;
    ~ActionsGuard() { ::posix_spawn_file_actions_destroy(a); }
  } guard{&actions};
  if (::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null",
                                         O_RDONLY, 0) != 0 ||
      ::posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                         O_WRONLY, 0) != 0 ||
      ::posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null",
                                         O_WRONLY, 0) != 0) {
    return false;
  }
  std::vector<std::string> args = {"ffmpeg",    "-y",        "-i",
                                   input,       "-c",        "copy",
                                   "-movflags", "faststart", output};
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = 0;
  if (::posix_spawnp(&pid, "ffmpeg", &actions, nullptr, argv.data(),
                     ::environ) != 0) {
    return false;  // no ffmpeg on PATH, or the spawn failed
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSeconds);
  int status = 0;
  while (true) {
    const pid_t done = ::waitpid(pid, &status, WNOHANG);
    if (done == pid) break;
    if (done < 0) return false;
    if (std::chrono::steady_clock::now() >= deadline) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return false;
  std::error_code ec;
  const auto size = std::filesystem::file_size(output, ec);
  return !ec && size > 0;
}

/**
 * Looks at the top level boxes of an mp4 and tells whether the `moov` box
 * (the index a player needs) comes before `mdat` (the media data), i.e. the
 * file is already "faststart". Only box headers are read, never the media.
 * nullopt when it cannot be told (not an mp4, truncated, no moov/mdat).
 */
inline std::optional<bool> isFaststart(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  in.seekg(0, std::ios::end);
  const uint64_t fileSize = static_cast<uint64_t>(in.tellg());
  uint64_t pos = 0;
  for (int boxes = 0; boxes < 64 && pos + 8 <= fileSize; ++boxes) {
    unsigned char header[16];
    in.clear();
    in.seekg(static_cast<std::streamoff>(pos));
    in.read(reinterpret_cast<char*>(header), 8);
    if (in.gcount() != 8) return std::nullopt;
    uint64_t size = (uint64_t{header[0]} << 24) | (uint64_t{header[1]} << 16) |
                    (uint64_t{header[2]} << 8) | uint64_t{header[3]};
    const std::string type(reinterpret_cast<char*>(header) + 4, 4);
    if (type == "moov") return true;   // index first
    if (type == "mdat") return false;  // media first
    if (size == 1) {                   // 64-bit size follows the type
      in.read(reinterpret_cast<char*>(header + 8), 8);
      if (in.gcount() != 8) return std::nullopt;
      size = 0;
      for (int i = 8; i < 16; ++i) size = (size << 8) | header[i];
    } else if (size == 0) {
      return std::nullopt;  // box runs to the end and is neither moov nor mdat
    }
    if (size < 8) return std::nullopt;
    pos += size;
  }
  return std::nullopt;
}

enum class FaststartResult {
  AlreadyFaststart,  // nothing to do, ffmpeg was not run
  Remuxed,           // rewritten with moov first
  Failed,            // original left untouched
};

/**
 * Makes `path` faststart, once, when the video is finished (Python's
 * ensure_faststart does this at every download instead). If the file already
 * has `moov` before `mdat` (the normal case: the Python VideoManager encodes
 * with +faststart) ffmpeg is not run. Otherwise ffmpeg writes
 * `<path>.faststart.tmp.mp4` and only when that succeeds is it renamed over
 * the original. On any failure (no ffmpeg, not a valid mp4) the original is
 * left untouched, so the video is still served, just without faststart.
 */
inline FaststartResult faststartInPlace(
    const std::string& path, int timeoutSeconds = kFaststartTimeoutSeconds) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec))
    return FaststartResult::Failed;
  if (isFaststart(path) == std::optional<bool>(true)) {
    return FaststartResult::AlreadyFaststart;
  }
  const std::string tmp = path + ".faststart.tmp.mp4";
  if (!ensureFaststart(path, tmp, timeoutSeconds)) {
    std::filesystem::remove(tmp, ec);
    return FaststartResult::Failed;
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return FaststartResult::Failed;
  }
  return FaststartResult::Remuxed;
}

}  // namespace tt::utils
