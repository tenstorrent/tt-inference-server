// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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

  /** Stops warmup IPC listener, then kills worker processes. Call only after
   * any threads that dereference worker() pointers (e.g. result consumers) have
   *  finished. */
  void stop();

  bool isReady() const { return ready.load(); }
  size_t numWorkers() const { return workerCount; }

  std::vector<WorkerInfo> getWorkerInfo() const;

  SingleProcessWorker* worker(size_t idx);

  /** Returns false if the worker process has exited. Updates is_alive flag. */
  bool checkWorkerAlive(size_t workerIdx);

  /** Re-fork a crashed worker process and update the workers entry. */
  void restartWorker(size_t workerIdx);

 private:
  bool isWorkerWarmed(int workerId) const;

  void startWorkers();
  void startWarmupListener();
  void stopWarmupListener();
  void stopProcesses();
  void startLivenessChecker();
  void stopLivenessChecker();
  WorkerConfig makeWorkerConfig(int workerId);

  /** Parent: fork/exec worker subprocess; sets worker.pid to child pid. Does
   * not return in the child process. */
  pid_t startWorker(SingleProcessWorker& worker);

  size_t workerCount;

  std::vector<std::unique_ptr<SingleProcessWorker>> workers;

  std::unique_ptr<tt::ipc::IWarmupSignalQueue> warmupQueue;
  std::thread warmupListenerThread;
  std::thread livenessCheckerThread;
  std::atomic<bool> livenessCheckerShouldStop{false};
  std::mutex warmupMutex;
  std::condition_variable warmupCv;
  std::atomic<bool> warmupReceived{false};
  std::atomic<bool> ready{false};
  mutable std::mutex warmedMutex;
  mutable std::set<int> warmedWorkerIds;
};

/** Build a WorkerConfig for the worker subprocess (used by main.cpp --worker).
 */
WorkerConfig makeWorkerConfigForProcess(int workerId);

}  // namespace tt::worker
