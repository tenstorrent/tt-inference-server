// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ipc/warmup_signal_queue.hpp"
#include "worker/single_process_worker.hpp"

namespace tt::worker {

/**
 * Manages worker process lifecycle: spawning, warmup signaling, crash detection
 * and restart.
 *
 * Validates configuration at construction time -- throws std::invalid_argument
 * if numWorkers < 1 with a message guiding the operator to fix DEVICE_IDS.
 */
class WorkerManager {
 public:
  using WarmupQueueFactory =
      std::function<std::unique_ptr<tt::ipc::IWarmupSignalQueue>(
          const std::string& name, size_t capacity)>;

  WorkerManager(size_t numWorkers, std::string warmupQueueName,
                WarmupQueueFactory warmupFactory);
  ~WorkerManager();

  WorkerManager(const WorkerManager&) = delete;
  WorkerManager& operator=(const WorkerManager&) = delete;

  void start();

  /** Full stop: warmup listener + worker processes. */
  void stop();

  /** Stop only the warmup listener.
   *  Call this before joining consumer threads that hold pointers to workers. */
  void stopWarmupListener();

  /** Kill worker processes and release their resources.
   *  Call this only after all consumer threads using worker pointers have
   * joined. */
  void stopProcesses();

  bool isReady() const { return is_ready_.load(); }
  size_t numWorkers() const { return num_workers_; }
  bool isWorkerWarmed(int workerId) const;

  SingleProcessWorker* worker(size_t idx);

  /** Returns false if the worker process has exited. Updates is_alive flag. */
  bool checkWorkerAlive(size_t workerIdx);

  /** Re-fork a crashed worker process and update the workers_ entry. */
  void restartWorker(size_t workerIdx);

 private:
  void startWorkers();
  void startWarmupListenerThread();
  void waitForFirstWarmup();
  WorkerConfig makeWorkerConfig(int workerId);

  size_t num_workers_;
  std::string warmup_queue_name_;
  WarmupQueueFactory warmup_factory_;

  std::vector<std::unique_ptr<SingleProcessWorker>> workers_;

  std::unique_ptr<tt::ipc::IWarmupSignalQueue> warmup_queue_;
  std::thread warmup_listener_thread_;
  std::mutex warmup_mutex_;
  std::condition_variable warmup_cv_;
  std::atomic<bool> warmup_received_{false};
  std::atomic<bool> is_ready_{false};
  mutable std::mutex warmed_mutex_;
  mutable std::set<int> warmed_worker_ids_;
};

/** Build a WorkerConfig for the worker subprocess (used by main.cpp --worker).
 */
WorkerConfig makeWorkerConfigForProcess(int workerId);

}  // namespace tt::worker
