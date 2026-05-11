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
  std::lock_guard<std::mutex> lock(warmup_mutex_);
  if (warmup_thread_.joinable()) return;
  TT_LOG_INFO("[ImageService] Starting (runner={})", runner_->runnerType());
  warmup_thread_ = std::thread([this] {
    try {
      if (!runner_->warmup()) {
        TT_LOG_ERROR("[ImageService] Runner warmup failed");
        return;
      }
      ready_.store(true, std::memory_order_release);
      TT_LOG_INFO("[ImageService] Started");
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[ImageService] Warmup threw: {}", e.what());
    } catch (...) {
      TT_LOG_ERROR("[ImageService] Warmup threw unknown exception");
    }
  });
}

void ImageService::stop() {
  std::thread t;
  {
    std::lock_guard<std::mutex> lock(warmup_mutex_);
    t = std::move(warmup_thread_);
  }
  if (t.joinable()) t.join();
  const bool wasReady = ready_.exchange(false, std::memory_order_acq_rel);
  if (wasReady && runner_) runner_->stop();
  if (wasReady) TT_LOG_INFO("[ImageService] Stopped");
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

domain::image::ImageResponse ImageService::processRequest(
    domain::ImageGenerateRequest request) {
  domain::image::ImageResponse response(request.task_id);
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
