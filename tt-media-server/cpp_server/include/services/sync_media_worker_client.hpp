// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <atomic>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ipc/file_payload_ipc.hpp"
#include "runtime/worker/worker_info.hpp"
#include "runtime/worker/worker_manager.hpp"

namespace tt::services {

struct SyncMediaWorkerResponse {
  Json::Value body;
  double generation_time_seconds = 0.0;
  std::string error;
};

class SyncMediaWorkerClient {
 public:
  SyncMediaWorkerClient(
      std::string serviceName,
      std::unique_ptr<tt::worker::WorkerManager> workerManager,
      std::unique_ptr<tt::ipc::file_payload::FilePayloadQueueManager>
          queueManager);
  ~SyncMediaWorkerClient();

  SyncMediaWorkerClient(const SyncMediaWorkerClient&) = delete;
  SyncMediaWorkerClient& operator=(const SyncMediaWorkerClient&) = delete;

  void start();
  void stop();

  bool isReady() const;
  size_t numWorkers() const;
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const;

  SyncMediaWorkerResponse submit(uint32_t taskId, const Json::Value& request);

 private:
  void startConsumers();
  void consumerLoopForWorker(size_t workerIdx);

  std::string payloadPath(uint32_t taskId, const char* prefix) const;

  std::string service_name_;
  std::unique_ptr<tt::worker::WorkerManager> worker_manager_;
  std::unique_ptr<tt::ipc::file_payload::FilePayloadQueueManager>
      queue_manager_;
  std::vector<std::thread> consumer_threads_;
  mutable std::mutex pending_mutex_;
  std::unordered_map<
      uint32_t,
      std::shared_ptr<
          std::promise<tt::ipc::file_payload::FilePayloadResult>>>
      pending_results_;
  std::filesystem::path payload_dir_;
  std::atomic<bool> running_{false};
};

}  // namespace tt::services
