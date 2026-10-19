// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/image_service.hpp"

#include <json/json.h>

#include <string>
#include <utility>

#include "config/settings.hpp"
#include "runtime/worker/worker_info.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {

domain::image::ImageResponse responseFromJson(uint32_t taskId,
                                              const Json::Value& json) {
  domain::image::ImageResponse response(taskId);
  if (json.isMember("images") && json["images"].isArray()) {
    for (const auto& image : json["images"]) {
      response.images.push_back(image.asString());
    }
  }
  if (json.isMember("generation_time")) {
    response.generation_time_seconds = json["generation_time"].asDouble();
  }
  if (json.isMember("error")) {
    response.error = json["error"].asString();
  }
  return response;
}

}  // namespace

ImageService::ImageService(
    config::ImageConfig config,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet> queueManager)
    : imageConfig(std::move(config)),
      workerScheduler(std::make_unique<MediaWorkerScheduler>(
          "image", std::move(workerManager), std::move(queueManager))) {
  this->maxQueueSize = tt::config::maxQueueSize();
  TT_LOG_INFO(
      "[ImageService] Initialized worker-backed image service "
      "(workers={}, max_queue_size={})",
      workerScheduler->numWorkers(), this->maxQueueSize);
}

ImageService::~ImageService() { stop(); }

void ImageService::start() {
  TT_LOG_INFO("[ImageService] Starting worker-backed service (workers={})",
              workerScheduler->numWorkers());
  workerScheduler->start();
}

void ImageService::stop() { workerScheduler->stop(); }

bool ImageService::isModelReady() const { return workerScheduler->isReady(); }

std::string ImageService::runnerInUse() const {
  return config::toClientRunnerName(imageConfig.runner_type);
}

std::vector<tt::worker::WorkerInfo> ImageService::getWorkerInfo() const {
  return workerScheduler->getWorkerInfo();
}

void ImageService::preProcess(domain::ImageGenerateRequest& /*request*/) const {
  const size_t prev = inFlight.fetch_add(1, std::memory_order_acq_rel);
  if (prev >= this->maxQueueSize) {
    inFlight.fetch_sub(1, std::memory_order_acq_rel);
    throw QueueFullException{};
  }
}

size_t ImageService::currentQueueSize() const {
  return inFlight.load(std::memory_order_acquire);
}

void ImageService::produceResponseAsync(
    domain::ImageGenerateRequest request,
    std::function<void(domain::image::ImageResponse&&)> onComplete) {
  const uint32_t taskId = request.task_id;
  if (!workerScheduler->isReady()) {
    inFlight.fetch_sub(1, std::memory_order_acq_rel);
    domain::image::ImageResponse response(taskId);
    response.error = "Image service not ready";
    onComplete(std::move(response));
    return;
  }

  workerScheduler->submitAsync(
      taskId, request.toJson(),
      [this, taskId,
       onComplete = std::move(onComplete)](MediaWorkerResult&& workerResponse) {
        inFlight.fetch_sub(1, std::memory_order_acq_rel);
        domain::image::ImageResponse response(taskId);
        response.generation_time_seconds =
            workerResponse.generation_time_seconds;
        if (!workerResponse.error.empty()) {
          response.error = std::move(workerResponse.error);
          onComplete(std::move(response));
          return;
        }
        auto parsed = responseFromJson(taskId, workerResponse.body);
        parsed.generation_time_seconds = workerResponse.generation_time_seconds;
        onComplete(std::move(parsed));
      });
}

}  // namespace tt::services
