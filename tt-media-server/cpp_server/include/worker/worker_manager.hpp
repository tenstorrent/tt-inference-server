// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "worker/single_process_worker.hpp"
#include "worker/worker_info.hpp"

namespace tt::ipc {
class IWarmupSignalQueue;
}

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
  explicit WorkerManager(size_t numWorkers);
  ~WorkerManager();

  WorkerManager(const WorkerManager&) = delete;
  WorkerManager& operator=(const WorkerManager&) = delete;

  void start();

  /** Stops warmup IPC listener, then kills worker processes. Call only after any
   *  threads that dereference worker() pointers (e.g. result consumers) have
   *  finished. */
  void stop();

  bool isReady() const { return is_ready_.load(); }
  size_t numWorkers() const { return num_workers_; }

  std::vector<WorkerInfo> getWorkerInfo() const;

  SingleProcessWorker* worker(size_t idx);

  /** Returns false if the worker process has exited. Updates is_alive flag. */
  bool checkWorkerAlive(size_t workerIdx);

  /** Re-fork a crashed worker process and update the workers_ entry. */
  void restartWorker(size_t workerIdx);

 private:
  bool isWorkerWarmed(int workerId) const;

  void startWorkers();
  void startWarmupListenerThread();
  void waitForFirstWarmup();
  void stopWarmupListener();
  void stopProcesses();
  WorkerConfig makeWorkerConfig(int workerId);

  /** Parent: fork/exec worker subprocess; sets worker.pid to child pid. Does not
   *  return in the child process. */
  pid_t startWorker(SingleProcessWorker& worker);

  size_t num_workers_;

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
