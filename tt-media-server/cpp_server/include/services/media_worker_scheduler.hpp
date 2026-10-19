// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ipc/media_payload_ipc.hpp"
#include "runtime/worker/worker_info.hpp"
#include "runtime/worker/worker_manager.hpp"

namespace tt::services {

struct MediaWorkerResult {
  Json::Value body;
  double generation_time_seconds = 0.0;
  std::string error;
};

/** Parent-side scheduler for media workers.
 *
 * Owns worker lifecycle, async request dispatch, result collection, and the
 * temp-file payload handoff used for large media responses.
 */
class MediaWorkerScheduler {
 public:
  using ResultCallback = std::function<void(MediaWorkerResult&&)>;

  MediaWorkerScheduler(
      std::string serviceName,
      std::unique_ptr<tt::worker::WorkerManager> workerManager,
      std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet> queueSet);
  ~MediaWorkerScheduler();

  MediaWorkerScheduler(const MediaWorkerScheduler&) = delete;
  MediaWorkerScheduler& operator=(const MediaWorkerScheduler&) = delete;

  void start();
  void stop();

  bool isReady() const;
  size_t numWorkers() const;
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const;

  void submitAsync(uint32_t taskId, const Json::Value& request,
                   ResultCallback onComplete);

 private:
  struct PendingEntry {
    ResultCallback callback;
    std::string requestPath;
    std::string responsePath;
    int claimedByWorker = -1;
  };

  void startConsumers();
  void consumerLoopForWorker(size_t workerIdx);
  void onWorkerDied(size_t workerIdx);
  void drainPending(const std::string& error);
  static void removePayloadFiles(const std::string& requestPath,
                                 const std::string& responsePath);

  std::string payloadPath(uint32_t taskId, const char* prefix) const;

  std::string service_name_;
  std::unique_ptr<tt::worker::WorkerManager> worker_manager_;
  std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet> queue_set_;
  std::vector<std::thread> consumer_threads_;
  mutable std::mutex pending_mutex_;
  std::unordered_map<uint32_t, PendingEntry> pending_results_;
  std::filesystem::path payload_dir_;
  std::atomic<bool> running_{false};
};

}  // namespace tt::services
