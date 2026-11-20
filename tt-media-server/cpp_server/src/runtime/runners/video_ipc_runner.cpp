// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/video_ipc_runner.hpp"

#include <stdexcept>
#include <utility>

#include "domain/video/video_response.hpp"
#include "runtime/runners/runner_registry.hpp"
#include "utils/logger.hpp"
#include "utils/video_faststart.hpp"

namespace tt::runners {

VideoIpcRunner::VideoIpcRunner(config::VideoConfig config, int workerId)
    : MediaIpcRunner("VideoIpcRunner", workerId),
      videoConfig(std::move(config)) {}

VideoIpcRunner::~VideoIpcRunner() { stop(); }

bool VideoIpcRunner::warmup() {
  runner = tt::utils::RunnerRegistry::instance().createMedia<MediaRunner>(
      config::ModelService::VIDEO, videoConfig.runner_type,
      config::RunnerConfig{videoConfig});
  if (!runner) {
    throw std::runtime_error(
        "[VideoIpcRunner] no media runner registered for runner_type=" +
        config::toString(videoConfig.runner_type));
  }
  TT_LOG_INFO("[VideoIpcRunner] Worker {} warming media runner ({})",
              workerId(), runner->runnerType());
  return runner->warmup();
}

void VideoIpcRunner::stop() {
  if (runner) {
    runner->stop();
  }
  MediaIpcRunner::stop();
}

Json::Value VideoIpcRunner::processJsonTask(const Json::Value& requestJson,
                                            uint32_t taskId) {
  auto request =
      tt::domain::VideoGenerateRequest::fromJson(requestJson, taskId);

  tt::domain::video::VideoResponse response(taskId);
  response.videoPath = runner->run(request);
  // Make the finished mp4 faststart once, here, so /download can stream the
  // file as it is. Skipped when it already is (the usual case); a failure
  // keeps the original file (it still plays).
  const std::string& path = response.videoPath;
  if (path.size() > 4 && path.rfind("mock://", 0) != 0 &&
      path.compare(path.size() - 4, 4, ".mp4") == 0) {
    switch (tt::utils::faststartInPlace(path)) {
      case tt::utils::FaststartResult::AlreadyFaststart:
        TT_LOG_INFO("[VideoIpcRunner] task {} mp4 already faststart", taskId);
        break;
      case tt::utils::FaststartResult::Remuxed:
        TT_LOG_INFO("[VideoIpcRunner] task {} faststart remux done", taskId);
        break;
      case tt::utils::FaststartResult::Failed:
        TT_LOG_WARN(
            "[VideoIpcRunner] task {} faststart remux failed, keeping original",
            taskId);
        break;
    }
  }
  return response.toOpenaiJson();
}

}  // namespace tt::runners
