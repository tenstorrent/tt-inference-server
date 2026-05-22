// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/image_service.hpp"

#include <json/json.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>

#include "config/settings.hpp"
#include "domain/image/image_request_json.hpp"
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

void writeJsonFile(const std::filesystem::path& path, const Json::Value& json) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open image IPC payload file: " +
                             path.string());
  }
  output << Json::writeString(builder, json);
}

Json::Value readJsonFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open image IPC response file: " +
                             path.string());
  }
  Json::CharReaderBuilder builder;
  Json::Value json;
  std::string errors;
  if (!Json::parseFromStream(builder, input, &json, &errors)) {
    throw std::runtime_error("failed to parse image IPC response file: " +
                             errors);
  }
  return json;
}

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

ImageService::ImageService(config::ImageConfig config,
                           std::unique_ptr<Runner> runner)
    : ImageService(std::move(config), singleRunner(std::move(runner))) {}

ImageService::ImageService(config::ImageConfig config, RunnerList runners)
    : config_(std::move(config)),
      runners_(std::move(runners)),
      runner_in_flight_(runners_.size()) {
  if (runners_.empty()) {
    throw std::invalid_argument(
        "[ImageService] at least one runner is required");
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

ImageService::ImageService(
    config::ImageConfig config,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::image::ImageQueueManager> queueManager)
    : config_(std::move(config)),
      worker_manager_(std::move(workerManager)),
      image_queue_manager_(std::move(queueManager)) {
  if (!worker_manager_) {
    throw std::invalid_argument("[ImageService] workerManager must not be null");
  }
  if (!image_queue_manager_) {
    throw std::invalid_argument("[ImageService] queueManager must not be null");
  }
  this->maxQueueSize = tt::config::maxQueueSize();
  ipc_payload_dir_ =
      std::filesystem::temp_directory_path() /
      ("tt-image-ipc-" + std::to_string(::getpid()));
  std::filesystem::create_directories(ipc_payload_dir_);
  TT_LOG_INFO("[ImageService] Initialized worker-backed image service "
              "(workers={}, max_queue_size={}, payload_dir='{}')",
              worker_manager_->numWorkers(), this->maxQueueSize,
              ipc_payload_dir_.string());
}

ImageService::~ImageService() { stop(); }

void ImageService::start() {
  if (worker_manager_) {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    TT_LOG_INFO("[ImageService] Starting worker-backed service (workers={})",
                worker_manager_->numWorkers());
    worker_manager_->start();
    startWorkerConsumers();
    return;
  }

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
  if (worker_manager_) {
    stopWorkerMode();
    return;
  }

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
  if (worker_manager_) {
    return worker_manager_->isReady();
  }
  return ready_.load(std::memory_order_acquire);
}

std::string ImageService::runnerInUse() const {
  return config::toClientRunnerName(config_.runner_type);
}

std::vector<tt::worker::WorkerInfo> ImageService::getWorkerInfo() const {
  if (worker_manager_) {
    return worker_manager_->getWorkerInfo();
  }
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

void ImageService::startWorkerConsumers() {
  const size_t n = worker_manager_->numWorkers();
  consumer_threads_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    consumer_threads_.emplace_back(&ImageService::consumerLoopForWorker, this,
                                   i);
  }
}

void ImageService::consumerLoopForWorker(size_t workerIdx) {
  auto resultQueue = image_queue_manager_->resultQueues.at(workerIdx);
  TT_LOG_INFO("[ImageService] Image consumer {} started", workerIdx);
  while (running_.load(std::memory_order_acquire)) {
    tt::ipc::image::ImageResult result;
    if (!resultQueue->blockingPop(result)) {
      break;
    }

    std::shared_ptr<std::promise<tt::ipc::image::ImageResult>> promise;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto it = pending_results_.find(result.task_id);
      if (it != pending_results_.end()) {
        promise = std::move(it->second);
        pending_results_.erase(it);
      }
    }
    if (promise) {
      promise->set_value(std::move(result));
    } else {
      TT_LOG_WARN("[ImageService] Dropping image result for unknown task {}",
                  result.task_id);
    }
  }
  TT_LOG_INFO("[ImageService] Image consumer {} stopped", workerIdx);
}

void ImageService::stopWorkerMode() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  TT_LOG_INFO("[ImageService] Stopping worker-backed service");

  for (size_t i = 0; i < worker_manager_->numWorkers(); ++i) {
    image_queue_manager_->taskQueue->push(tt::ipc::image::ImageTask::done());
  }

  for (auto& queue : image_queue_manager_->resultQueues) {
    queue->shutdown();
  }

  for (auto& thread : consumer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumer_threads_.clear();

  worker_manager_->stop();
  image_queue_manager_->clear();

  std::error_code ec;
  std::filesystem::remove_all(ipc_payload_dir_, ec);
  TT_LOG_INFO("[ImageService] Worker-backed service stopped");
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

  if (worker_manager_) {
    return processWorkerRequest(request);
  }

  return processInProcessRequest(request);
}

domain::image::ImageResponse ImageService::processInProcessRequest(
    const domain::ImageGenerateRequest& request) {
  domain::image::ImageResponse response(request.task_id);
  if (!ready_.load(std::memory_order_acquire)) {
    response.error = "Image service not ready";
    return response;
  }

  const auto t0 = std::chrono::steady_clock::now();
  try {
    const size_t runnerIndex = selectRunnerIndex();
    struct RunnerInFlightGuard {
      std::atomic<size_t>& counter;
      ~RunnerInFlightGuard() {
        counter.fetch_sub(1, std::memory_order_acq_rel);
      }
    } runnerGuard{runner_in_flight_[runnerIndex]};
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

domain::image::ImageResponse ImageService::processWorkerRequest(
    const domain::ImageGenerateRequest& request) {
  domain::image::ImageResponse response(request.task_id);
  if (!worker_manager_->isReady()) {
    response.error = "Image service not ready";
    return response;
  }

  const auto requestPath =
      ipc_payload_dir_ /
      ("request-" + std::to_string(request.task_id) + ".json");
  const auto responsePath =
      ipc_payload_dir_ /
      ("response-" + std::to_string(request.task_id) + ".json");

  auto promise = std::make_shared<
      std::promise<tt::ipc::image::ImageResult>>();
  auto future = promise->get_future();
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_results_[request.task_id] = promise;
  }

  try {
    writeJsonFile(requestPath, domain::image::toJson(request));

    tt::ipc::image::ImageTask task;
    task.task_id = request.task_id;
    task.request_path = requestPath.string();
    task.response_path = responsePath.string();
    image_queue_manager_->taskQueue->push(task);

    auto result = future.get();
    response.generation_time_seconds = result.generation_time_seconds;
    if (!result.error.empty()) {
      response.error = std::move(result.error);
    } else {
      response = responseFromJson(request.task_id, readJsonFile(responsePath));
    }
  } catch (const std::exception& e) {
    response.error = e.what();
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_results_.erase(request.task_id);
  }

  std::error_code ec;
  std::filesystem::remove(requestPath, ec);
  std::filesystem::remove(responsePath, ec);
  return response;
}

}  // namespace tt::services
