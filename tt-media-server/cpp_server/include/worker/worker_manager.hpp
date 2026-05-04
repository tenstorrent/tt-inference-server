// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <functional>
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
/**
 * Result of polling a worker process for liveness.  Returned by
 * pollProcessLiveness() so callers can react to the alive -> dead transition
 * exactly once.
 */
struct ProcessLivenessTransition {
  bool stillAlive;          ///< True if the process is still running.
  bool transitionedToDead;  ///< True iff this call observed alive -> dead.
};

/**
 * Reap the worker process if it has exited and update `aliveFlag` accordingly.
 * Logs the cause of death (signal / exit code) at the appropriate severity.
 *
 * Pure with respect to WorkerManager state: callers wire the death callback
 * separately based on the returned transition flag.  Extracted so that the
 * waitpid + bookkeeping logic can be unit-tested without spinning up a real
 * worker subprocess.
 */
ProcessLivenessTransition pollProcessLiveness(pid_t pid,
                                              std::atomic<bool>& aliveFlag,
                                              size_t workerIdx);

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

  /** Returns false if the worker process has exited. Updates is_alive flag.
   * Fires the worker-death callback (if set) exactly once on the
   * alive -> dead transition. */
  bool checkWorkerAlive(size_t workerIdx);

  /** Re-fork a crashed worker process and update the workers entry. */
  void restartWorker(size_t workerIdx);

  /** Callback fired when a worker is detected dead (exited or signaled).
   * Invoked once per worker, on the alive -> dead transition, from the
   * liveness checker thread (or from whichever thread calls
   * checkWorkerAlive). The callback must be lightweight and must not call
   * back into WorkerManager::stop(); deadlock would result. */
  using WorkerDeathCallback = std::function<void(size_t workerIdx, pid_t pid)>;
  void setWorkerDeathCallback(WorkerDeathCallback callback);

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

  mutable std::mutex deathCallbackMutex;
  WorkerDeathCallback deathCallback;
};

/** Build a WorkerConfig for the worker subprocess (used by main.cpp --worker).
 */
WorkerConfig makeWorkerConfigForProcess(int workerId);

}  // namespace tt::worker
