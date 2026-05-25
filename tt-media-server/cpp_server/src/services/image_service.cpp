// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/image_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "config/settings.hpp"
#include "runtime/worker/worker_info.hpp"
#include "utils/logger.hpp"

namespace tt::services {

ImageService::ImageService(config::ImageConfig config,
                           std::unique_ptr<Runner> runner)
    : config_(std::move(config)), runner_(std::move(runner)) {
  if (!runner_) {
    throw std::invalid_argument("[ImageService] runner must not be null");
  }
  this->maxQueueSize = tt::config::maxQueueSize();
  TT_LOG_INFO("[ImageService] Initialized with runner '{}' (max_queue_size={})",
              runner_->runnerType(), this->maxQueueSize);
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
  if (runner_) runner_->stop();
  if (wasReady) TT_LOG_INFO("[ImageService] Stopped");
}

bool ImageService::isModelReady() const {
  return ready_.load(std::memory_order_acquire);
}

std::string ImageService::runnerInUse() const {
  return config::toClientRunnerName(config_.runner_type);
}

std::vector<tt::worker::WorkerInfo> ImageService::getWorkerInfo() const {
  const bool ready = ready_.load(std::memory_order_acquire);
  const size_t count = std::max<size_t>(1, tt::config::numWorkers());
  std::vector<tt::worker::WorkerInfo> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    tt::worker::WorkerInfo info;
    info.worker_id = std::to_string(i);
    info.is_ready = ready;
    info.is_alive = true;
    out.push_back(std::move(info));
  }
  return out;
}

void ImageService::preProcess(domain::ImageGenerateRequest& /*request*/) const {
  const size_t prev = in_flight_.fetch_add(1, std::memory_order_acq_rel);
  if (prev >= this->maxQueueSize) {
    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
    throw QueueFullException{};
  }
}

size_t ImageService::currentQueueSize() const {
  return in_flight_.load(std::memory_order_acquire);
}

domain::image::ImageResponse ImageService::produceResponse(
    domain::ImageGenerateRequest request) {
  struct InFlightGuard {
    std::atomic<size_t>& counter;
    ~InFlightGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
  } guard{in_flight_};

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
