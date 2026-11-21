// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/mock_video_runner.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>

#include "config/types.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

MockVideoRunner::MockVideoRunner(config::VideoConfig config)
    : videoConfig(std::move(config)) {}

bool MockVideoRunner::warmup() {
  if (const char* ms = std::getenv("TT_VIDEO_MOCK_WARMUP_MS"); ms && *ms) {
    const long wait = std::strtol(ms, nullptr, 10);
    if (wait > 0) std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }
  return true;
}

std::string MockVideoRunner::run(
    const tt::domain::VideoGenerateRequest& request) {
  TT_LOG_INFO(
      "[MockVideoRunner] task {} runner {} prompt_chars {} image_prompts {}",
      request.task_id, config::toString(videoConfig.runner_type),
      request.prompt.size(),
      request.imagePrompts ? request.imagePrompts->size() : 0);
  // Test hook: TT_VIDEO_MOCK_DELAY_MS makes the mock behave like a slow device
  // so queued / running / cancelled states can be exercised without a chip.
  if (const char* delay = std::getenv("TT_VIDEO_MOCK_DELAY_MS");
      delay && *delay) {
    const long ms = std::strtol(delay, nullptr, 10);
    if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }
  // Test hook: TT_VIDEO_MOCK_WRITE_FILE=1 returns a small real file (an mp4
  // header, not playable) instead of a mock:// reference, so download and the
  // sync response can be exercised without a chip. It goes to
  // TT_VIDEO_OUTPUT_DIR, or /tmp/tt-video-mock-out when that is not set.
  if (const char* flag = std::getenv("TT_VIDEO_MOCK_WRITE_FILE");
      flag && (std::string(flag) == "1" || std::string(flag) == "ffmpeg" ||
               std::string(flag) == "ffmpeg-faststart")) {
    const char* dirEnv = std::getenv("TT_VIDEO_OUTPUT_DIR");
    const std::filesystem::path dir =
        (dirEnv && *dirEnv) ? dirEnv : "/tmp/tt-video-mock-out";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path =
        dir / ("mock-" + std::to_string(request.task_id) + ".mp4");
    // "ffmpeg": a real, short mp4 with the moov atom last (ffmpeg's default),
    // so the worker's faststart remux can be checked end to end.
    // "ffmpeg-faststart": the same, but already faststart (what the real
    // encoder produces), so the worker must skip the remux.
    if (std::string(flag) != "1") {
      const std::string movflags = std::string(flag) == "ffmpeg-faststart"
                                       ? "-movflags +faststart "
                                       : "";
      const std::string cmd =
          "ffmpeg -y -loglevel error -f lavfi -i testsrc=size=64x64:rate=5 "
          "-t 1 -pix_fmt yuv420p " +
          movflags + "'" + path.string() + "'";
      if (std::system(cmd.c_str()) == 0) return path.string();
    }
    std::ofstream out(path, std::ios::binary);
    const unsigned char header[] = {0,   0,   0,   0x18, 'f', 't', 'y', 'p',
                                    'i', 's', 'o', 'm',  0,   0,   2,   0,
                                    'i', 's', 'o', 'm',  'm', 'p', '4', '1'};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out << "mock video payload for task " << request.task_id;
    return path.string();
  }
  return "mock://video/" + std::to_string(request.task_id);
}

}  // namespace tt::runners
