// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/image_service.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "utils/logger.hpp"
#include "worker/worker_info.hpp"

namespace tt::services {

ImageService::ImageService(config::ImageConfig config,
                           std::unique_ptr<Runner> runner)
    : config_(std::move(config)), runner_(std::move(runner)) {
  if (!runner_) {
    throw std::invalid_argument("[ImageService] runner must not be null");
  }
  TT_LOG_INFO("[ImageService] Initialized with runner '{}'",
              runner_->runnerType());
}

ImageService::~ImageService() { stop(); }

void ImageService::start() {
  TT_LOG_INFO("[ImageService] Starting (runner={})", runner_->runnerType());
  if (!runner_->warmup()) {
    throw std::runtime_error("[ImageService] Runner warmup failed");
  }
  ready_.store(true, std::memory_order_release);
  TT_LOG_INFO("[ImageService] Started");
}

void ImageService::stop() {
  if (!ready_.exchange(false, std::memory_order_acq_rel)) return;
  if (runner_) runner_->stop();
  TT_LOG_INFO("[ImageService] Stopped");
}

bool ImageService::isModelReady() const {
  return ready_.load(std::memory_order_acquire);
}

std::vector<tt::worker::WorkerInfo> ImageService::getWorkerInfo() const {
  tt::worker::WorkerInfo info;
  info.worker_id = "0";
  info.is_ready = ready_.load(std::memory_order_acquire);
  info.is_alive = true;
  return {info};
}

domain::ImageResponse ImageService::processRequest(
    domain::ImageGenerateRequest request) {
  domain::ImageResponse response(request.task_id);
  if (!ready_.load(std::memory_order_acquire)) {
    response.error = "Image service not ready";
    return response;
  }

  const auto t0 = std::chrono::steady_clock::now();
  try {
    response.images = runner_->run(request);
  } catch (const std::exception& e) {
    response.error = e.what();
    TT_LOG_ERROR("[ImageService] Runner threw: {}", e.what());
  }
  const auto t1 = std::chrono::steady_clock::now();
  response.generation_time_seconds =
      std::chrono::duration<double>(t1 - t0).count();
  return response;
}

}  // namespace tt::services
