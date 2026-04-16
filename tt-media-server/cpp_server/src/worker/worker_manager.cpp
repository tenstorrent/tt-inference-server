// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/worker_manager.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#include "config/settings.hpp"
#include "ipc/boost_ipc_cancel_queue.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "ipc/boost_ipc_warmup_signal_queue.hpp"
#include "ipc/queue_manager.hpp"
#include "utils/logger.hpp"

namespace {

[[noreturn]] void execWorkerProcessHelper(
    size_t workerId,
    const std::unordered_map<std::string, std::string>& envVars) {
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

}  // namespace

namespace tt::worker {

WorkerManager::WorkerManager(size_t numWorkers) : workerCount{numWorkers} {
  if (workerCount < 1) {
    throw std::invalid_argument(
        "WorkerManager requires at least 1 worker. "
        "Set DEVICE_IDS with at least one bracket pair, "
        "e.g. DEVICE_IDS=\"(0)\"");
  }
}

WorkerManager::~WorkerManager() { stop(); }

void WorkerManager::start() {
  startWarmupListener();
  startWorkers();
  startLivenessChecker();
}

void WorkerManager::stop() {
  stopLivenessChecker();
  stopWarmupListener();
  stopProcesses();
}

void WorkerManager::stopWarmupListener() {
  if (warmupQueue) {
    warmupQueue->remove();
    warmupQueue.reset();
  }
  if (warmupListenerThread.joinable()) {
    warmupListenerThread.join();
  }
  {
    std::lock_guard<std::mutex> lock(warmedMutex);
    warmedWorkerIds.clear();
  }
  ready = false;
}

void WorkerManager::stopProcesses() {
  for (auto& w : workers) {
    w->stop();
  }
  workers.clear();
}

bool WorkerManager::isWorkerWarmed(int workerId) const {
  std::lock_guard<std::mutex> lock(warmedMutex);
  return warmedWorkerIds.count(workerId) != 0;
}

std::vector<WorkerInfo> WorkerManager::getWorkerInfo() const {
  std::vector<WorkerInfo> out;
  out.reserve(workers.size());
  for (const auto& w : workers) {
    WorkerInfo info;
    info.worker_id = std::to_string(w->worker_id);
    info.pid = w->pid;
    info.is_alive = w->is_alive;
    // Worker is ready only if it's warmed up and the process is still alive
    info.is_ready = w->is_alive && isWorkerWarmed(w->worker_id);
    out.push_back(std::move(info));
  }
  return out;
}

SingleProcessWorker* WorkerManager::worker(size_t idx) {
  return workers[idx].get();
}

bool WorkerManager::checkWorkerAlive(size_t workerIdx) {
  auto* w = workers[workerIdx].get();
  if (w->pid <= 0) {
    return false;
  }
  int status;
  pid_t result = waitpid(w->pid, &status, WNOHANG);
  if (result == 0) {
    return true;
  }
  if (result == w->pid) {
    w->is_alive = false;
    if (WIFSIGNALED(status)) {
      int sig = WTERMSIG(status);
      TT_LOG_CRITICAL(
          "[WorkerManager] Worker {} (PID {}) killed by signal {} ({})"
          "{}",
          workerIdx, w->pid, sig, strsignal(sig),
          WCOREDUMP(status) ? " -- core dumped" : "");
    } else if (WIFEXITED(status)) {
      int exitCode = WEXITSTATUS(status);
      if (exitCode != 0) {
        TT_LOG_CRITICAL(
            "[WorkerManager] Worker {} (PID {}) exited with code {}", workerIdx,
            w->pid, exitCode);
      } else {
        TT_LOG_WARN(
            "[WorkerManager] Worker {} (PID {}) exited normally (code 0)",
            workerIdx, w->pid);
      }
    } else {
      TT_LOG_CRITICAL(
          "[WorkerManager] Worker {} (PID {}) terminated, raw status=0x{:x}",
          workerIdx, w->pid, status);
    }
    return false;
  }
  return true;  // waitpid error -- assume alive
}

void WorkerManager::restartWorker(size_t workerIdx) {
  TT_LOG_WARN("[WorkerManager] Restarting crashed worker {}", workerIdx);
  auto cfg = makeWorkerConfig(static_cast<int>(workerIdx));
  workers[workerIdx] = std::make_unique<SingleProcessWorker>(cfg);
  auto& w = workers[workerIdx];
  pid_t pid = startWorker(*w);
  TT_LOG_INFO("[WorkerManager] Restarted worker {} with PID {}", workerIdx,
              pid);
}

WorkerConfig WorkerManager::makeWorkerConfig(int workerId) {
  WorkerConfig cfg;
  cfg.env_vars["TT_VISIBLE_DEVICES"] =
      tt::config::visibleDevicesForWorker(workerId);
  cfg.task_queue = std::make_shared<tt::ipc::BoostIpcTaskQueue>(
      tt::config::ttTaskQueueName());
  cfg.result_queue = std::make_shared<tt::ipc::BoostIpcResultQueue>(
      std::string(tt::config::ttResultQueueName()) + std::to_string(workerId));
  cfg.cancel_queue = std::make_shared<tt::ipc::BoostIpcCancelQueue>(
      std::string(tt::config::ttCancelQueueName()) + std::to_string(workerId));
  cfg.worker_id = workerId;
  cfg.runner_config = tt::config::llmEngineConfig();
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
  worker.pid = pid;
  return pid;
}

void WorkerManager::startWorkers() {
  for (size_t i = 0; i < workerCount; ++i) {
    auto cfg = makeWorkerConfig(static_cast<int>(i));
    workers.push_back(std::make_unique<SingleProcessWorker>(cfg));
    pid_t pid = startWorker(*workers[i]);
    TT_LOG_INFO("[WorkerManager] Spawned worker {} with PID {}", i, pid);
  }
}

void WorkerManager::startWarmupListener() {
  warmupQueue = std::make_unique<tt::ipc::BoostIpcWarmupSignalQueue>(
      tt::ipc::WARMUP_SIGNALS_QUEUE_NAME, workerCount);
  warmupReceived = false;
  warmupListenerThread = std::thread([this]() {
    try {
      for (size_t i = 0; i < workerCount; ++i) {
        int workerId = warmupQueue->receive();
        TT_LOG_INFO("[WorkerManager] Worker {} warmed up", workerId);
        {
          std::lock_guard<std::mutex> lock(warmedMutex);
          warmedWorkerIds.insert(workerId);
        }
        if (i == 0) {
          ready = true;
          warmupReceived = true;
          warmupCv.notify_all();
        }
      }
    } catch (const std::exception& e) {
      TT_LOG_WARN("[WorkerManager] Warmup listener failed: {} (shutdown?)",
                  e.what());
    } catch (...) {
      TT_LOG_WARN("[WorkerManager] Warmup listener failed: unknown exception");
    }
  });
}

void WorkerManager::startLivenessChecker() {
  livenessCheckerShouldStop = false;
  livenessCheckerThread = std::thread([this]() {
    TT_LOG_INFO("[WorkerManager] Liveness checker thread started");
    while (!livenessCheckerShouldStop.load()) {
      // Wait 5 seconds, but wake up if stop is requested
      for (int i = 0; i < 50 && !livenessCheckerShouldStop.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      if (livenessCheckerShouldStop.load()) {
        break;
      }

      // Check all workers
      for (size_t i = 0; i < workers.size(); ++i) {
        bool alive = checkWorkerAlive(i);
        if (!alive) {
          TT_LOG_ERROR(
              "[WorkerManager] Liveness check: Worker {} is dead (PID {})", i,
              workers[i]->pid);
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
  cfg.env_vars["TT_VISIBLE_DEVICES"] =
      tt::config::visibleDevicesForWorker(workerId);
  cfg.task_queue = std::make_shared<tt::ipc::BoostIpcTaskQueue>(
      tt::config::ttTaskQueueName());
  cfg.result_queue = std::make_shared<tt::ipc::BoostIpcResultQueue>(
      std::string(tt::config::ttResultQueueName()) + std::to_string(workerId));
  cfg.cancel_queue = std::make_shared<tt::ipc::BoostIpcCancelQueue>(
      std::string(tt::config::ttCancelQueueName()) + std::to_string(workerId));
  cfg.worker_id = workerId;
  cfg.runner_config = tt::config::llmEngineConfig();
  return cfg;
}

}  // namespace tt::worker
