// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/image_ipc_runner.hpp"

#include <stdexcept>
#include <utility>

#include "domain/image/image_response.hpp"
#include "runtime/runners/runner_registry.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

ImageIpcRunner::ImageIpcRunner(config::ImageConfig config, int workerId)
    : MediaIpcRunner("ImageIpcRunner", workerId),
      config_(std::move(config)) {}

ImageIpcRunner::~ImageIpcRunner() { stop(); }

bool ImageIpcRunner::warmup() {
  runner_ = tt::utils::RunnerRegistry::instance().createMedia<MediaRunner>(
      config::ModelService::IMAGE, config_.runner_type,
      config::RunnerConfig{config_});
  if (!runner_) {
    throw std::runtime_error(
        "[ImageIpcRunner] no media runner registered for runner_type=" +
        config::toString(config_.runner_type));
  }
  TT_LOG_INFO("[ImageIpcRunner] Worker {} warming media runner ({})", workerId(),
              runner_->runnerType());
  return runner_->warmup();
}

void ImageIpcRunner::stop() {
  if (runner_) {
    runner_->stop();
  }
  MediaIpcRunner::stop();
}

Json::Value ImageIpcRunner::processJsonTask(const Json::Value& requestJson,
                                            uint32_t taskId) {
  auto request =
      tt::domain::ImageGenerateRequest::fromJson(requestJson, taskId);

  tt::domain::image::ImageResponse response(taskId);
  response.images = runner_->run(request);
  return response.toOpenaiJson();
}

}  // namespace tt::runners
