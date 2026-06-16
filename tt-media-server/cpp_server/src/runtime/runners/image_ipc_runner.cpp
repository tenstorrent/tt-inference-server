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
      imageConfig(std::move(config)) {}

ImageIpcRunner::~ImageIpcRunner() { stop(); }

bool ImageIpcRunner::warmup() {
  runner = tt::utils::RunnerRegistry::instance().createMedia<MediaRunner>(
      config::ModelService::IMAGE, imageConfig.runner_type,
      config::RunnerConfig{imageConfig});
  if (!runner) {
    throw std::runtime_error(
        "[ImageIpcRunner] no media runner registered for runner_type=" +
        config::toString(imageConfig.runner_type));
  }
  TT_LOG_INFO("[ImageIpcRunner] Worker {} warming media runner ({})",
              workerId(), runner->runnerType());
  return runner->warmup();
}

void ImageIpcRunner::stop() {
  if (runner) {
    runner->stop();
  }
  MediaIpcRunner::stop();
}

Json::Value ImageIpcRunner::processJsonTask(const Json::Value& requestJson,
                                            uint32_t taskId) {
  auto request =
      tt::domain::ImageGenerateRequest::fromJson(requestJson, taskId);

  tt::domain::image::ImageResponse response(taskId);
  response.images = runner->run(request);
  return response.toOpenaiJson();
}

}  // namespace tt::runners
