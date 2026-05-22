// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/sync_media_worker_client.hpp"

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

SyncMediaWorkerClient::SyncMediaWorkerClient(
    std::string serviceName,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::file_payload::FilePayloadQueueManager>
        queueManager)
    : service_name_(std::move(serviceName)),
      worker_manager_(std::move(workerManager)),
      queue_manager_(std::move(queueManager)) {
  if (!worker_manager_) {
    throw std::invalid_argument(
        "[SyncMediaWorkerClient] workerManager must not be null");
  }
  if (!queue_manager_) {
    throw std::invalid_argument(
        "[SyncMediaWorkerClient] queueManager must not be null");
  }
  payload_dir_ = std::filesystem::temp_directory_path() /
                 ("tt-" + service_name_ + "-ipc-" + std::to_string(::getpid()));
  std::filesystem::create_directories(payload_dir_);
  TT_LOG_INFO(
      "[SyncMediaWorkerClient] Initialized {} worker client "
      "(workers={}, payload_dir='{}')",
      service_name_, worker_manager_->numWorkers(), payload_dir_.string());
}

SyncMediaWorkerClient::~SyncMediaWorkerClient() { stop(); }

void SyncMediaWorkerClient::start() {
  if (running_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  TT_LOG_INFO("[SyncMediaWorkerClient] Starting {} worker client (workers={})",
              service_name_, worker_manager_->numWorkers());
  worker_manager_->start();
  startConsumers();
}

void SyncMediaWorkerClient::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  TT_LOG_INFO("[SyncMediaWorkerClient] Stopping {} worker client",
              service_name_);

  for (size_t i = 0; i < worker_manager_->numWorkers(); ++i) {
    queue_manager_->taskQueue->push(
        tt::ipc::file_payload::FilePayloadTask::done());
  }

  for (auto& queue : queue_manager_->resultQueues) {
    queue->shutdown();
  }

  for (auto& thread : consumer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumer_threads_.clear();

  worker_manager_->stop();
  queue_manager_->clear();

  std::error_code ec;
  std::filesystem::remove_all(payload_dir_, ec);
  TT_LOG_INFO("[SyncMediaWorkerClient] {} worker client stopped",
              service_name_);
}

bool SyncMediaWorkerClient::isReady() const {
  return worker_manager_->isReady();
}

size_t SyncMediaWorkerClient::numWorkers() const {
  return worker_manager_->numWorkers();
}

std::vector<tt::worker::WorkerInfo> SyncMediaWorkerClient::getWorkerInfo()
    const {
  return worker_manager_->getWorkerInfo();
}

SyncMediaWorkerResponse SyncMediaWorkerClient::submit(
    uint32_t taskId, const Json::Value& request) {
  SyncMediaWorkerResponse response;
  if (!worker_manager_->isReady()) {
    response.error = "Media service not ready";
    return response;
  }

  const auto requestPath = payloadPath(taskId, "request");
  const auto responsePath = payloadPath(taskId, "response");

  auto promise = std::make_shared<
      std::promise<tt::ipc::file_payload::FilePayloadResult>>();
  auto future = promise->get_future();
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_results_[taskId] = promise;
  }

  try {
    writeJsonFile(requestPath, request);

    tt::ipc::file_payload::FilePayloadTask task;
    task.task_id = taskId;
    task.request_path = requestPath;
    task.response_path = responsePath;
    queue_manager_->taskQueue->push(task);

    auto result = future.get();
    response.generation_time_seconds = result.generation_time_seconds;
    if (!result.error.empty()) {
      response.error = std::move(result.error);
    } else {
      response.body = readJsonFile(responsePath);
    }
  } catch (const std::exception& e) {
    response.error = e.what();
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_results_.erase(taskId);
  }

  std::error_code ec;
  std::filesystem::remove(requestPath, ec);
  std::filesystem::remove(responsePath, ec);
  return response;
}

void SyncMediaWorkerClient::startConsumers() {
  const size_t n = worker_manager_->numWorkers();
  consumer_threads_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    consumer_threads_.emplace_back(&SyncMediaWorkerClient::consumerLoopForWorker,
                                   this, i);
  }
}

void SyncMediaWorkerClient::consumerLoopForWorker(size_t workerIdx) {
  auto resultQueue = queue_manager_->resultQueues.at(workerIdx);
  TT_LOG_INFO("[SyncMediaWorkerClient] {} consumer {} started", service_name_,
              workerIdx);
  while (running_.load(std::memory_order_acquire)) {
    tt::ipc::file_payload::FilePayloadResult result;
    if (!resultQueue->blockingPop(result)) {
      break;
    }

    std::shared_ptr<
        std::promise<tt::ipc::file_payload::FilePayloadResult>>
        promise;
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
      TT_LOG_WARN(
          "[SyncMediaWorkerClient] Dropping {} result for unknown task {}",
          service_name_, result.task_id);
    }
  }
  TT_LOG_INFO("[SyncMediaWorkerClient] {} consumer {} stopped", service_name_,
              workerIdx);
}

std::string SyncMediaWorkerClient::payloadPath(uint32_t taskId,
                                               const char* prefix) const {
  return (payload_dir_ /
          (std::string(prefix) + "-" + std::to_string(taskId) + ".json"))
      .string();
}

}  // namespace tt::services
