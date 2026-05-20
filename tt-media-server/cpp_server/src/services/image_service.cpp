// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/image_service.hpp"

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "config/settings.hpp"
#include "runtime/worker/worker_info.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {

ImageService::RunnerList singleRunner(
    std::unique_ptr<ImageService::Runner> runner) {
  ImageService::RunnerList runners;
  runners.push_back(std::move(runner));
  return runners;
}

}  // namespace

ImageService::ImageService(config::ImageConfig config,
                           std::unique_ptr<Runner> runner)
    : ImageService(std::move(config), singleRunner(std::move(runner))) {}

ImageService::ImageService(config::ImageConfig config, RunnerList runners)
    : config_(std::move(config)),
      runners_(std::move(runners)),
      runner_in_flight_(runners_.size()) {
  if (runners_.empty()) {
    throw std::invalid_argument("[ImageService] at least one runner is required");
  }
  for (const auto& runner : runners_) {
    if (!runner) {
      throw std::invalid_argument("[ImageService] runner must not be null");
    }
  }
  this->maxQueueSize = tt::config::maxQueueSize();
  TT_LOG_INFO(
      "[ImageService] Initialized with {} runner(s), primary runner '{}' "
      "(max_queue_size={})",
      runners_.size(), runners_.front()->runnerType(), this->maxQueueSize);
}

ImageService::~ImageService() { stop(); }

void ImageService::start() {
  std::lock_guard<std::mutex> lock(warmup_mutex_);
  if (warmup_thread_.joinable()) return;
  TT_LOG_INFO("[ImageService] Starting (runners={})", runners_.size());
  warmup_thread_ = std::thread([this] {
    try {
      for (size_t i = 0; i < runners_.size(); ++i) {
        TT_LOG_INFO("[ImageService] Warming runner {}/{} ({})", i + 1,
                    runners_.size(), runners_[i]->runnerType());
        if (!runners_[i]->warmup()) {
          TT_LOG_ERROR("[ImageService] Runner {}/{} warmup failed", i + 1,
                       runners_.size());
          for (auto& runner : runners_) runner->stop();
          return;
        }
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
  for (auto& runner : runners_) runner->stop();
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
  const size_t count = runners_.size();
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

size_t ImageService::selectRunnerIndex() const {
  const size_t count = runners_.size();
  const size_t start =
      next_runner_.fetch_add(1, std::memory_order_relaxed) % count;
  size_t best = start;
  size_t bestLoad = runner_in_flight_[best].load(std::memory_order_acquire);
  for (size_t offset = 1; offset < count; ++offset) {
    const size_t idx = (start + offset) % count;
    const size_t load = runner_in_flight_[idx].load(std::memory_order_acquire);
    if (load < bestLoad) {
      best = idx;
      bestLoad = load;
      if (bestLoad == 0) break;
    }
  }
  return best;
}

domain::image::ImageResponse ImageService::processRequest(
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
    const size_t runnerIndex = selectRunnerIndex();
    InFlightGuard runnerGuard{runner_in_flight_[runnerIndex]};
    runner_in_flight_[runnerIndex].fetch_add(1, std::memory_order_acq_rel);
    response.images = runners_[runnerIndex]->run(request);
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
