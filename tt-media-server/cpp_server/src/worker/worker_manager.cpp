// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/worker_manager.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "config/settings.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "ipc/boost_ipc_warmup_signal_queue.hpp"
#include "ipc/queue_manager.hpp"
#include "ipc/shared_memory.hpp"
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

WorkerManager::WorkerManager(size_t numWorkers) : num_workers_{numWorkers} {
  if (num_workers_ < 1) {
    throw std::invalid_argument(
        "WorkerManager requires at least 1 worker. "
        "Set DEVICE_IDS with at least one bracket pair, "
        "e.g. DEVICE_IDS=\"(0)\"");
  }
}

WorkerManager::~WorkerManager() { stop(); }

void WorkerManager::start() {
  startWarmupListenerThread();
  startWorkers();
  waitForFirstWarmup();
}

void WorkerManager::stop() {
  stopWarmupListener();
  stopProcesses();
}

void WorkerManager::stopWarmupListener() {
  if (warmup_queue_) {
    warmup_queue_->remove();
    warmup_queue_.reset();
  }
  if (warmup_listener_thread_.joinable()) {
    warmup_listener_thread_.join();
  }
  {
    std::lock_guard<std::mutex> lock(warmed_mutex_);
    warmed_worker_ids_.clear();
  }
  is_ready_ = false;
}

void WorkerManager::stopProcesses() {
  for (auto& w : workers_) {
    w->stop();
  }
  workers_.clear();
}

bool WorkerManager::isWorkerWarmed(int workerId) const {
  std::lock_guard<std::mutex> lock(warmed_mutex_);
  return warmed_worker_ids_.count(workerId) != 0;
}

std::vector<WorkerInfo> WorkerManager::getWorkerInfo() const {
  std::vector<WorkerInfo> out;
  out.reserve(workers_.size());
  for (const auto& w : workers_) {
    WorkerInfo info;
    info.worker_id = std::to_string(w->worker_id);
    info.is_ready = isWorkerWarmed(w->worker_id);
    out.push_back(std::move(info));
  }
  return out;
}

SingleProcessWorker* WorkerManager::worker(size_t idx) {
  return workers_[idx].get();
}

bool WorkerManager::checkWorkerAlive(size_t workerIdx) {
  auto* w = workers_[workerIdx].get();
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
    return false;
  }
  return true;  // waitpid error -- assume alive
}

void WorkerManager::restartWorker(size_t workerIdx) {
  TT_LOG_WARN("[WorkerManager] Restarting crashed worker {}", workerIdx);
  auto cfg = makeWorkerConfig(static_cast<int>(workerIdx));
  workers_[workerIdx] = std::make_unique<SingleProcessWorker>(cfg);
  auto& w = workers_[workerIdx];
  pid_t pid = startWorker(*w);
  TT_LOG_INFO("[WorkerManager] Restarted worker {} with PID {}", workerIdx,
              pid);
}

WorkerConfig WorkerManager::makeWorkerConfig(int workerId) {
  WorkerConfig cfg;
  cfg.env_vars["TT_VISIBLE_DEVICES"] =
      tt::config::visibleDevicesForWorker(workerId);
  cfg.task_queue =
      std::make_shared<tt::ipc::BoostIpcTaskQueue>(tt::ipc::TASK_QUEUE_NAME);
  cfg.result_queue =
      std::make_shared<tt::ipc::TokenRingBuffer<tt::ipc::RING_BUFFER_CAPACITY>>(
          "/tt_tokens_" + std::to_string(workerId), false);
  cfg.worker_id = workerId;
  cfg.runner_config = tt::config::llmEngineConfig();
  return cfg;
}

pid_t WorkerManager::startWorker(SingleProcessWorker& worker) {
  const size_t slot = static_cast<size_t>(worker.worker_id);
  pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("Failed to fork worker process");
  }
  if (pid == 0) {
    setpgid(0, 0);
    execWorkerProcessHelper(slot, worker.cfg.env_vars);
  }
  setpgid(pid, pid);
  worker.pid = pid;
  return pid;
}

void WorkerManager::startWorkers() {
  for (size_t i = 0; i < num_workers_; ++i) {
    auto cfg = makeWorkerConfig(static_cast<int>(i));
    workers_.push_back(std::make_unique<SingleProcessWorker>(cfg));
    pid_t pid = startWorker(*workers_[i]);
    TT_LOG_INFO("[WorkerManager] Spawned worker {} with PID {}", i, pid);
  }
}

void WorkerManager::startWarmupListenerThread() {
  const char* name = tt::ipc::WARMUP_SIGNALS_QUEUE_NAME;
  tt::ipc::BoostIpcWarmupSignalQueue::remove(name);
  warmup_queue_ =
      std::make_unique<tt::ipc::BoostIpcWarmupSignalQueue>(name, num_workers_);
  warmup_received_ = false;
  warmup_listener_thread_ = std::thread([this]() {
    try {
      for (size_t i = 0; i < num_workers_; ++i) {
        int workerId = warmup_queue_->receive();
        TT_LOG_INFO("[WorkerManager] Worker {} warmed up", workerId);
        {
          std::lock_guard<std::mutex> lock(warmed_mutex_);
          warmed_worker_ids_.insert(workerId);
        }
        if (i == 0) {
          is_ready_ = true;
          warmup_received_ = true;
          warmup_cv_.notify_all();
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

void WorkerManager::waitForFirstWarmup() {
  if (!warmup_queue_) return;
  std::unique_lock<std::mutex> lock(warmup_mutex_);
  warmup_cv_.wait(lock, [this]() { return warmup_received_.load(); });
}

WorkerConfig makeWorkerConfigForProcess(int workerId) {
  WorkerConfig cfg;
  cfg.env_vars["TT_VISIBLE_DEVICES"] =
      tt::config::visibleDevicesForWorker(workerId);
  cfg.task_queue =
      std::make_shared<tt::ipc::BoostIpcTaskQueue>(tt::ipc::TASK_QUEUE_NAME);
  cfg.result_queue =
      std::make_shared<tt::ipc::TokenRingBuffer<tt::ipc::RING_BUFFER_CAPACITY>>(
          "/tt_tokens_" + std::to_string(workerId), false);
  cfg.worker_id = workerId;
  cfg.runner_config = tt::config::llmEngineConfig();
  return cfg;
}

}  // namespace tt::worker
