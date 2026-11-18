// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/worker/worker_manager.hpp"

#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#include "config/settings.hpp"
#include "ipc/boost/boost_cancel_queue.hpp"
#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/boost/boost_task_queue.hpp"
#include "ipc/boost/boost_warmup_signal_queue.hpp"
#include "ipc/tts_ipc.hpp"
#include "utils/logger.hpp"

namespace {

[[noreturn]] void execWorkerProcessHelper(
    size_t workerId,
    const std::unordered_map<std::string, std::string>& envVars) {
  // Request SIGTERM when the parent (main server) process dies so that workers
  // do not become orphaned under PID 1 if the main process crashes.  This
  // survives execv, so it only needs to be set once here in the child.
  prctl(PR_SET_PDEATHSIG, SIGTERM);

  for (const auto& [key, value] : envVars) {
    setenv(key.c_str(), value.c_str(), 1);
  }
  char exePath[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
  if (n <= 0) {
    perror("readlink /proc/self/exe");
    _exit(1);
  }
  exePath[n] = '\0';
  char idBuf[16];
  std::snprintf(idBuf, sizeof(idBuf), "%zu", workerId);
  char* execArgv[] = {exePath, const_cast<char*>("--worker"), idBuf, nullptr};
  execv(exePath, execArgv);
  perror("execv");
  _exit(1);
}

/** Sentinel for awaitWorkersWarmup(): wait for warmup/exit with no deadline. */
constexpr unsigned NO_WARMUP_DEADLINE = 0;

unsigned defaultWarmupTimeoutMs() {
  if (tt::config::isEmbeddingService()) {
    return tt::config::embeddingWarmupTimeoutMs();
  }
  if (tt::config::isImageService()) {
    if (const char* env = std::getenv("WARMUP_TIMEOUT_MS"); env && *env) {
      return tt::config::warmupTimeoutMs();
    }
    return tt::config::imageEngineConfig()
               .weights_distribution_timeout_seconds *
           1000u;
  }
  // LLM workers: no supervisor-side deadline (unchanged from before the
  // embedding/image warmup coordination). The Blaze runners enforce their own
  // budgets (PM_CONNECT_TIMEOUT_MS for connect, WARMUP_TIMEOUT_MS per
  // ALLOCATE / first-token / EVICT phase) and exit on failure, which the
  // coordinator detects via checkWorkerAlive().
  return NO_WARMUP_DEADLINE;
}

void populateCommonWorkerEnv(tt::worker::WorkerConfig& cfg, int workerId) {
  const std::string visibleDevices =
      tt::config::visibleDevicesForWorker(static_cast<size_t>(workerId));
  cfg.env_vars["TT_WORKER_ID"] = std::to_string(workerId);
  cfg.env_vars["TT_VISIBLE_DEVICES"] = visibleDevices;

  if (tt::config::isEmbeddingService() || tt::config::isImageService()) {
    cfg.env_vars["OMP_NUM_THREADS"] = "2";
    cfg.env_vars["MKL_NUM_THREADS"] = "2";
    cfg.env_vars["TORCH_NUM_THREADS"] = "1";

    if (const char* throttle = std::getenv("DEFAULT_THROTTLE_LEVEL");
        throttle && *throttle) {
      cfg.env_vars["TT_MM_THROTTLE_PERF"] = throttle;
    }

    if (const char* metalHome = std::getenv("TT_METAL_HOME");
        metalHome && *metalHome) {
      std::string deviceSuffix = visibleDevices;
      std::replace(deviceSuffix.begin(), deviceSuffix.end(), ',', '_');
      cfg.env_vars["TT_METAL_CACHE"] =
          std::string(metalHome) + "/built/" + deviceSuffix;
    }
  }
}

}  // namespace

namespace tt::worker {

namespace {

void configureLlmQueues(WorkerConfig& cfg, int workerId) {
  cfg.task_queue = std::make_shared<tt::ipc::boost::TaskQueue>(
      tt::config::ttTaskQueueName());
  cfg.result_queue = std::make_shared<tt::ipc::boost::ResultQueue>(
      std::string(tt::config::ttResultQueueName()) + std::to_string(workerId));
  cfg.cancel_queue = std::make_shared<tt::ipc::boost::CancelQueue>(
      std::string(tt::config::ttCancelQueueName()) + std::to_string(workerId));
}

void configureTtsQueues(WorkerConfig& cfg, int workerId) {
  tt::ipc::tts::TtsQueueNames names;
  cfg.task_queue =
      std::make_shared<tt::ipc::tts::TtsTaskQueue>(names.taskQueue);
  cfg.result_queue = std::make_shared<tt::ipc::tts::TtsAudioChunkQueue>(
      names.audioQueuePrefix + std::to_string(workerId));
  cfg.cancel_queue = std::make_shared<tt::ipc::boost::CancelQueue>(
      names.cancelQueuePrefix + std::to_string(workerId));
}

}  // namespace

WorkerManager::WorkerManager(size_t numWorkers, WorkerManagerOptions options)
    : workerCount{numWorkers}, options_{options} {
  if (workerCount < 1) {
    throw std::invalid_argument(
        "WorkerManager requires at least 1 worker. "
        "Set DEVICE_IDS with at least one bracket pair, "
        "e.g. DEVICE_IDS=\"(0)\"");
  }
}

WorkerManager::~WorkerManager() { stop(); }

void WorkerManager::start() {
  warmupQueue = std::make_unique<tt::ipc::boost::WarmupSignalQueue>(
      tt::config::ttWarmupSignalsQueueName(), workerCount);
  warmupReceived = false;
  ready = false;
  warmupShouldStop = false;

  {
    std::lock_guard<std::mutex> lock(workersMutex);
    workers.clear();
    workers.reserve(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
      auto cfg = makeWorkerConfig(static_cast<int>(i));
      workers.push_back(std::make_unique<SingleProcessWorker>(cfg));
      workers[i]->pid = -1;
      workers[i]->is_alive = false;
      workers[i]->is_ready = false;
    }
  }

  // Spawn worker 0 on the calling thread so PR_SET_PDEATHSIG is bound to the
  // main server thread for the leader. When sequentialLeadWorkerWarmup is
  // disabled (or there is only 1 worker), spawn all workers immediately.
  const size_t initialSpawnCount =
      (workerCount > 1 && options_.sequentialLeadWorkerWarmup) ? 1
                                                               : workerCount;
  for (size_t i = 0; i < initialSpawnCount; ++i) {
    pid_t pid = startWorker(*workers[i]);
    TT_LOG_INFO("[WorkerManager] Spawned worker {} with PID {}", i, pid);
  }

  startWarmupListener();
  startLivenessChecker();
}

void WorkerManager::stop() {
  stopLivenessChecker();
  stopWarmupListener();
  stopProcesses();
}

void WorkerManager::stopWarmupListener() {
  warmupShouldStop.store(true, std::memory_order_release);
  if (warmupListenerThread.joinable()) {
    warmupListenerThread.join();
  }
  if (warmupQueue) {
    warmupQueue->remove();
    warmupQueue.reset();
  }
  {
    std::lock_guard<std::mutex> lock(warmedMutex);
    warmedWorkerIds.clear();
  }
  ready = false;
}

void WorkerManager::stopProcesses() {
  std::lock_guard<std::mutex> lock(workersMutex);
  for (auto& w : workers) {
    w->stop();
  }
  workers.clear();
}

bool WorkerManager::isWorkerWarmed(int workerId) const {
  std::lock_guard<std::mutex> lock(warmedMutex);
  return warmedWorkerIds.contains(workerId);
}

bool WorkerManager::isReady() const {
  if (!ready.load(std::memory_order_acquire)) {
    return false;
  }
  std::lock_guard<std::mutex> wLock(workersMutex);
  std::lock_guard<std::mutex> mLock(warmedMutex);
  bool anyReady = false;
  for (const auto& w : workers) {
    if (!w) continue;
    if (w->pid > 0 && !w->is_alive.load(std::memory_order_acquire)) {
      return false;
    }
    if (w->is_alive.load(std::memory_order_acquire) &&
        warmedWorkerIds.contains(w->worker_id)) {
      anyReady = true;
    }
  }
  return anyReady;
}

std::vector<WorkerInfo> WorkerManager::getWorkerInfo() const {
  std::lock_guard<std::mutex> lock(workersMutex);
  std::vector<WorkerInfo> out;
  out.reserve(workers.size());
  for (const auto& w : workers) {
    WorkerInfo info;
    info.worker_id = std::to_string(w->worker_id);
    info.pid = w->pid;
    info.is_alive = w->is_alive.load(std::memory_order_acquire);
    // Worker is ready only if it's warmed up and the process is still alive
    info.is_ready = info.is_alive && isWorkerWarmed(w->worker_id);
    out.push_back(std::move(info));
  }
  return out;
}

SingleProcessWorker* WorkerManager::worker(size_t idx) {
  std::lock_guard<std::mutex> lock(workersMutex);
  return workers[idx].get();
}

void WorkerManager::notifyWorkerDeath(size_t workerIdx, pid_t pid) {
  WorkerDeathCallback callback;
  {
    std::lock_guard<std::mutex> lock(deathCallbackMutex);
    callback = deathCallback;
  }
  if (callback) {
    try {
      callback(workerIdx, pid);
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[WorkerManager] Worker death callback threw: {}", e.what());
    } catch (...) {
      TT_LOG_ERROR(
          "[WorkerManager] Worker death callback threw a non-std exception");
    }
  }
}

void WorkerManager::stopSpawnedWorkers() {
  for (size_t i = 0; i < workerCount; ++i) {
    SingleProcessWorker* w = worker(i);
    if (!w || w->pid <= 0) continue;
    w->stop();
    if (w->is_alive.exchange(false, std::memory_order_acq_rel)) {
      notifyWorkerDeath(i, w->pid);
    }
  }
}

bool WorkerManager::checkWorkerAlive(size_t workerIdx) {
  SingleProcessWorker* w = nullptr;
  {
    std::lock_guard<std::mutex> lock(workersMutex);
    if (workerIdx >= workers.size()) return false;
    w = workers[workerIdx].get();
  }
  if (!w || w->pid <= 0) {
    return false;
  }
  auto result = pollProcessLiveness(w->pid, w->is_alive, workerIdx);
  if (result.transitionedToDead) {
    notifyWorkerDeath(workerIdx, w->pid);
  }
  return result.stillAlive;
}

void WorkerManager::setWorkerDeathCallback(WorkerDeathCallback callback) {
  std::lock_guard<std::mutex> lock(deathCallbackMutex);
  deathCallback = std::move(callback);
}

void WorkerManager::restartWorker(size_t workerIdx) {
  TT_LOG_WARN("[WorkerManager] Restarting crashed worker {}", workerIdx);
  auto cfg = makeWorkerConfig(static_cast<int>(workerIdx));
  std::lock_guard<std::mutex> lock(workersMutex);
  workers[workerIdx] = std::make_unique<SingleProcessWorker>(cfg);
  auto& w = workers[workerIdx];
  pid_t pid = startWorker(*w);
  TT_LOG_INFO("[WorkerManager] Restarted worker {} with PID {}", workerIdx,
              pid);
}

WorkerConfig WorkerManager::makeWorkerConfig(int workerId) {
  WorkerConfig cfg;
  populateCommonWorkerEnv(cfg, workerId);
  if (tt::config::isTtsService()) {
    configureTtsQueues(cfg, workerId);
  } else if (!tt::config::isImageService() &&
             !tt::config::isEmbeddingService()) {
    configureLlmQueues(cfg, workerId);
  }
  cfg.worker_id = workerId;
  cfg.runner_config =
      tt::config::workerRunnerConfig(static_cast<size_t>(workerId));
  return cfg;
}

pid_t WorkerManager::startWorker(SingleProcessWorker& worker) {
  pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("Failed to fork worker process");
  }
  if (pid == 0) {
    setpgid(0, 0);
    execWorkerProcessHelper(static_cast<size_t>(worker.worker_id),
                            worker.cfg.env_vars);
  }
  setpgid(pid, pid);
  worker.pid.store(pid, std::memory_order_release);
  worker.is_alive.store(true, std::memory_order_release);
  return pid;
}

bool WorkerManager::awaitWorkersWarmup(std::vector<size_t> pending,
                                       unsigned timeoutMs) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

  while (!pending.empty() &&
         !warmupShouldStop.load(std::memory_order_acquire)) {
    int wid = -1;
    while (warmupQueue && warmupQueue->tryReceive(wid)) {
      TT_LOG_INFO("[WorkerManager] Worker {} warmed up", wid);
      {
        std::lock_guard<std::mutex> lock(warmedMutex);
        warmedWorkerIds.insert(wid);
      }
      ready.store(true, std::memory_order_release);
      warmupReceived.store(true, std::memory_order_release);
      warmupCv.notify_all();
      std::erase(pending, static_cast<size_t>(wid));
    }

    std::vector<size_t> stillPending;
    stillPending.reserve(pending.size());
    for (size_t idx : pending) {
      if (isWorkerWarmed(static_cast<int>(idx))) {
        continue;
      }
      if (!checkWorkerAlive(idx)) {
        TT_LOG_ERROR("[WorkerManager] Worker {} exited during warmup", idx);
        ready.store(false, std::memory_order_release);
        return false;
      }
      stillPending.push_back(idx);
    }
    pending = std::move(stillPending);
    if (pending.empty()) {
      return true;
    }

    if (timeoutMs != NO_WARMUP_DEADLINE &&
        std::chrono::steady_clock::now() >= deadline) {
      for (size_t idx : pending) {
        TT_LOG_ERROR(
            "[WorkerManager] Worker {} timed out in warmup after {}ms; "
            "stopping it",
            idx, timeoutMs);
        SingleProcessWorker* w = worker(idx);
        if (w) {
          w->stop();
          if (w->is_alive.exchange(false, std::memory_order_acq_rel)) {
            notifyWorkerDeath(idx, w->pid);
          }
        }
      }
      ready.store(false, std::memory_order_release);
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return pending.empty();
}

void WorkerManager::runWarmupCoordinator(size_t firstUncoordinatedIdx,
                                         unsigned timeoutMs) {
  try {
    if (firstUncoordinatedIdx == 1 && workerCount > 1) {
      // Leader-first warmup: worker 0 was already spawned on start().
      if (!awaitWorkersWarmup({0}, timeoutMs) ||
          warmupShouldStop.load(std::memory_order_acquire)) {
        TT_LOG_ERROR(
            "[WorkerManager] Leader worker 0 failed warmup; aborting remaining "
            "worker startup");
        stopSpawnedWorkers();
        ready.store(false, std::memory_order_release);
        return;
      }

      std::vector<size_t> remaining;
      for (size_t i = 1; i < workerCount; ++i) {
        if (warmupShouldStop.load(std::memory_order_acquire)) return;
        SingleProcessWorker* w = worker(i);
        if (!w) continue;
        pid_t pid = startWorker(*w);
        TT_LOG_INFO("[WorkerManager] Spawned worker {} with PID {}", i, pid);
        remaining.push_back(i);
      }
      if (!remaining.empty() &&
          !awaitWorkersWarmup(std::move(remaining), timeoutMs)) {
        TT_LOG_ERROR(
            "[WorkerManager] Worker warmup failed; stopping all spawned "
            "workers");
        stopSpawnedWorkers();
        ready.store(false, std::memory_order_release);
        return;
      }
    } else {
      std::vector<size_t> all;
      all.reserve(workerCount);
      for (size_t i = 0; i < workerCount; ++i) {
        all.push_back(i);
      }
      if (!awaitWorkersWarmup(std::move(all), timeoutMs)) {
        TT_LOG_ERROR(
            "[WorkerManager] Worker warmup failed; stopping all spawned "
            "workers");
        stopSpawnedWorkers();
        ready.store(false, std::memory_order_release);
        return;
      }
    }

    // Keep draining any late warmup signals until stop() is called so that
    // restarted workers (`restartWorker`) also get marked warmed.
    while (!warmupShouldStop.load(std::memory_order_acquire)) {
      int wid = -1;
      while (warmupQueue && warmupQueue->tryReceive(wid)) {
        TT_LOG_INFO("[WorkerManager] Worker {} warmed up", wid);
        {
          std::lock_guard<std::mutex> lock(warmedMutex);
          warmedWorkerIds.insert(wid);
        }
        ready.store(true, std::memory_order_release);
        warmupReceived.store(true, std::memory_order_release);
        warmupCv.notify_all();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  } catch (const std::exception& e) {
    TT_LOG_WARN("[WorkerManager] Warmup coordinator failed: {}", e.what());
  } catch (...) {
    TT_LOG_WARN("[WorkerManager] Warmup coordinator failed: unknown exception");
  }
}

void WorkerManager::startWarmupListener() {
  const unsigned timeoutMs = options_.warmupTimeoutMs > 0
                                 ? options_.warmupTimeoutMs
                                 : defaultWarmupTimeoutMs();
  const size_t firstUncoordinatedIdx =
      (workerCount > 1 && options_.sequentialLeadWorkerWarmup) ? 1
                                                               : workerCount;
  warmupListenerThread = std::thread(&WorkerManager::runWarmupCoordinator, this,
                                     firstUncoordinatedIdx, timeoutMs);
}

void WorkerManager::startLivenessChecker() {
  livenessCheckerShouldStop = false;
  livenessCheckerThread = std::thread([this]() {
    TT_LOG_INFO("[WorkerManager] Liveness checker thread started");
    while (!livenessCheckerShouldStop.load()) {
      for (int i = 0; i < 10 && !livenessCheckerShouldStop.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      if (livenessCheckerShouldStop.load()) {
        break;
      }

      for (size_t i = 0; i < workerCount; ++i) {
        SingleProcessWorker* w = worker(i);
        if (!w || w->pid <= 0 || !w->is_alive.load(std::memory_order_acquire)) {
          continue;
        }
        bool alive = checkWorkerAlive(i);
        if (!alive) {
          TT_LOG_ERROR(
              "[WorkerManager] Liveness check: Worker {} is dead (PID {})", i,
              w->pid.load(std::memory_order_acquire));
        }
      }
    }
    TT_LOG_INFO("[WorkerManager] Liveness checker thread stopped");
  });
}

void WorkerManager::stopLivenessChecker() {
  livenessCheckerShouldStop = true;
  if (livenessCheckerThread.joinable()) {
    livenessCheckerThread.join();
  }
}

WorkerConfig makeWorkerConfigForProcess(int workerId) {
  WorkerConfig cfg;
  populateCommonWorkerEnv(cfg, workerId);
  if (tt::config::isTtsService()) {
    configureTtsQueues(cfg, workerId);
  } else if (!tt::config::isImageService() &&
             !tt::config::isEmbeddingService()) {
    configureLlmQueues(cfg, workerId);
  }
  cfg.worker_id = workerId;
  cfg.runner_config =
      tt::config::workerRunnerConfig(static_cast<size_t>(workerId));
  return cfg;
}

}  // namespace tt::worker
