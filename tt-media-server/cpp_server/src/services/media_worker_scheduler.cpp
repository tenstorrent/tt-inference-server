// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/media_worker_scheduler.hpp"

#include <unistd.h>

#include <fstream>
#include <stdexcept>
#include <utility>

#include "utils/logger.hpp"

namespace tt::services {

namespace {

void writeJsonFile(const std::filesystem::path& path, const Json::Value& json) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open media IPC payload file: " +
                             path.string());
  }
  output << Json::writeString(builder, json);
}

Json::Value readJsonFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open media IPC response file: " +
                             path.string());
  }
  Json::CharReaderBuilder builder;
  Json::Value json;
  std::string errors;
  if (!Json::parseFromStream(builder, input, &json, &errors)) {
    throw std::runtime_error("failed to parse media IPC response file: " +
                             errors);
  }
  return json;
}

}  // namespace

MediaWorkerScheduler::MediaWorkerScheduler(
    std::string serviceName,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet> queueSet)
    : service_name_(std::move(serviceName)),
      worker_manager_(std::move(workerManager)),
      queue_set_(std::move(queueSet)) {
  if (!worker_manager_) {
    throw std::invalid_argument(
        "[MediaWorkerScheduler] workerManager must not be null");
  }
  if (!queue_set_) {
    throw std::invalid_argument(
        "[MediaWorkerScheduler] queueSet must not be null");
  }
  payload_dir_ = std::filesystem::temp_directory_path() /
                 ("tt-" + service_name_ + "-ipc-" + std::to_string(::getpid()));
  std::filesystem::create_directories(payload_dir_);
  TT_LOG_INFO(
      "[MediaWorkerScheduler] Initialized {} scheduler "
      "(workers={}, payload_dir='{}')",
      service_name_, worker_manager_->numWorkers(), payload_dir_.string());
}

MediaWorkerScheduler::~MediaWorkerScheduler() { stop(); }

void MediaWorkerScheduler::start() {
  if (running_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  TT_LOG_INFO("[MediaWorkerScheduler] Starting {} scheduler (workers={})",
              service_name_, worker_manager_->numWorkers());
  worker_manager_->setWorkerDeathCallback(
      [this](size_t workerIdx, pid_t /*pid*/) { onWorkerDied(workerIdx); });
  worker_manager_->start();
  startConsumers();
}

void MediaWorkerScheduler::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  TT_LOG_INFO("[MediaWorkerScheduler] Stopping {} scheduler", service_name_);

  for (size_t i = 0; i < worker_manager_->numWorkers(); ++i) {
    queue_set_->taskQueue->push(
        tt::ipc::media_payload::MediaPayloadTask::done());
  }

  for (auto& queue : queue_set_->resultQueues) {
    queue->shutdown();
  }

  for (auto& thread : consumer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumer_threads_.clear();

  worker_manager_->stop();
  drainPending("Server shutting down");
  queue_set_->clear();

  std::error_code ec;
  std::filesystem::remove_all(payload_dir_, ec);
  TT_LOG_INFO("[MediaWorkerScheduler] {} scheduler stopped", service_name_);
}

bool MediaWorkerScheduler::isReady() const {
  return worker_manager_->isReady();
}

size_t MediaWorkerScheduler::numWorkers() const {
  return worker_manager_->numWorkers();
}

std::vector<tt::worker::WorkerInfo> MediaWorkerScheduler::getWorkerInfo()
    const {
  return worker_manager_->getWorkerInfo();
}

void MediaWorkerScheduler::removePayloadFiles(const std::string& requestPath,
                                              const std::string& responsePath) {
  std::error_code ec;
  std::filesystem::remove(requestPath, ec);
  std::filesystem::remove(responsePath, ec);
}

void MediaWorkerScheduler::submitAsync(uint32_t taskId,
                                       const Json::Value& request,
                                       ResultCallback onComplete) {
  if (!running_.load(std::memory_order_acquire) ||
      !worker_manager_->isReady()) {
    MediaWorkerResult err;
    err.error = "Media service not ready";
    onComplete(std::move(err));
    return;
  }

  const auto requestPath = payloadPath(taskId, "request");
  const auto responsePath = payloadPath(taskId, "response");

  try {
    writeJsonFile(requestPath, request);
  } catch (const std::exception& e) {
    removePayloadFiles(requestPath, responsePath);
    MediaWorkerResult err;
    err.error = e.what();
    onComplete(std::move(err));
    return;
  }

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_results_.emplace(
        taskId,
        PendingEntry{std::move(onComplete), requestPath, responsePath, -1});
  }

  try {
    tt::ipc::media_payload::MediaPayloadTask task;
    task.task_id = taskId;
    task.request_path = requestPath;
    task.response_path = responsePath;
    queue_set_->taskQueue->push(task);
  } catch (const std::exception& e) {
    ResultCallback cb;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto it = pending_results_.find(taskId);
      if (it != pending_results_.end()) {
        cb = std::move(it->second.callback);
        pending_results_.erase(it);
      }
    }
    removePayloadFiles(requestPath, responsePath);
    if (cb) {
      MediaWorkerResult err;
      err.error = e.what();
      cb(std::move(err));
    }
  }
}

void MediaWorkerScheduler::startConsumers() {
  const size_t n = worker_manager_->numWorkers();
  consumer_threads_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    consumer_threads_.emplace_back(&MediaWorkerScheduler::consumerLoopForWorker,
                                   this, i);
  }
}

void MediaWorkerScheduler::consumerLoopForWorker(size_t workerIdx) {
  auto resultQueue = queue_set_->resultQueues.at(workerIdx);
  TT_LOG_INFO("[MediaWorkerScheduler] {} consumer {} started", service_name_,
              workerIdx);
  while (running_.load(std::memory_order_acquire)) {
    tt::ipc::media_payload::MediaPayloadResult result;
    if (!resultQueue->blockingPop(result)) {
      break;
    }

    if (result.isClaimed()) {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto it = pending_results_.find(result.task_id);
      if (it != pending_results_.end()) {
        it->second.claimedByWorker = static_cast<int>(workerIdx);
      }
      continue;
    }

    PendingEntry entry;
    bool found = false;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto it = pending_results_.find(result.task_id);
      if (it != pending_results_.end()) {
        entry = std::move(it->second);
        pending_results_.erase(it);
        found = true;
      }
    }
    if (!found) {
      TT_LOG_WARN(
          "[MediaWorkerScheduler] Dropping {} result for unknown task {}",
          service_name_, result.task_id);
      continue;
    }

    MediaWorkerResult response;
    response.generation_time_seconds = result.generation_time_seconds;
    if (!result.error.empty()) {
      response.error = std::move(result.error);
    } else {
      try {
        response.body = readJsonFile(entry.responsePath);
      } catch (const std::exception& e) {
        response.error = e.what();
      }
    }
    removePayloadFiles(entry.requestPath, entry.responsePath);
    entry.callback(std::move(response));
  }
  TT_LOG_INFO("[MediaWorkerScheduler] {} consumer {} stopped", service_name_,
              workerIdx);
}

void MediaWorkerScheduler::onWorkerDied(size_t workerIdx) {
  const bool anyReady = worker_manager_->isReady();
  std::vector<PendingEntry> failed;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    for (auto it = pending_results_.begin(); it != pending_results_.end();) {
      if (!anyReady ||
          it->second.claimedByWorker == static_cast<int>(workerIdx)) {
        failed.push_back(std::move(it->second));
        it = pending_results_.erase(it);
      } else {
        ++it;
      }
    }
  }

  for (auto& entry : failed) {
    removePayloadFiles(entry.requestPath, entry.responsePath);
    MediaWorkerResult err;
    err.error = "Worker process crashed";
    entry.callback(std::move(err));
  }
}

void MediaWorkerScheduler::drainPending(const std::string& error) {
  std::vector<PendingEntry> drained;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    drained.reserve(pending_results_.size());
    for (auto& [taskId, entry] : pending_results_) {
      drained.push_back(std::move(entry));
    }
    pending_results_.clear();
  }
  for (auto& entry : drained) {
    removePayloadFiles(entry.requestPath, entry.responsePath);
    MediaWorkerResult err;
    err.error = error;
    entry.callback(std::move(err));
  }
}

std::string MediaWorkerScheduler::payloadPath(uint32_t taskId,
                                              const char* prefix) const {
  return (payload_dir_ /
          (std::string(prefix) + "-" + std::to_string(taskId) + ".json"))
      .string();
}

}  // namespace tt::services
