#include "worker/single_process_worker.hpp"

#include <sys/wait.h>

#include <chrono>
#include <csignal>
#include <thread>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "ipc/boost_ipc_warmup_signal_queue.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/runner_factory.hpp"

namespace tt::worker {

SingleProcessWorker::SingleProcessWorker(WorkerConfig& cfg)
    : cfg(std::move(cfg)) {
  pid = getpid();
  worker_id = cfg.worker_id;
  is_ready = true;
}

SingleProcessWorker::~SingleProcessWorker() = default;

void SingleProcessWorker::start() {
  tracy_config::tracySetThreadName(
      ("Worker-" + std::to_string(cfg.worker_id)).c_str());

  for (const auto& [key, value] : cfg.env_vars) {
    setenv(key.c_str(), value.c_str(), 1);
  }

  {
    ZoneScopedN("Worker::init");
    runner_ = tt::utils::runner_factory::createRunner(
        tt::config::modelService(), cfg.runner_config, cfg.result_queue.get(),
        cfg.task_queue.get(), cfg.cancel_queue.get());
  }
  TT_LOG_INFO(
      "[SingleProcessWorker] Worker {} starting runner (warmup may take a "
      "while)",
      worker_id);
  runner_->start([this]() {
    try {
      tt::ipc::BoostIpcWarmupSignalQueue warmupQueue(
          tt::ipc::WARMUP_SIGNALS_QUEUE_NAME);
      warmupQueue.sendReady(worker_id);
      TT_LOG_INFO("[SingleProcessWorker] Worker {} signaled warmup complete",
                  worker_id);
    } catch (const std::exception& e) {
      TT_LOG_ERROR(
          "[SingleProcessWorker] Worker {} failed to signal warmup: {}",
          worker_id, e.what());
    }
  });
}

void SingleProcessWorker::stop() {
  ZoneScopedN("Worker::stop");
  if (runner_) {
    runner_->stop();
  }
  if (pid > 0) {
    killpg(pid, SIGTERM);

    int status;
    int waitResult = waitpid(pid, &status, WNOHANG);
    if (waitResult == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(
          tt::config::defaults::WORKER_STOP_TIMEOUT_MS));
      waitResult = waitpid(pid, &status, WNOHANG);
      if (waitResult == 0) {
        killpg(pid, SIGKILL);
        waitpid(pid, &status, 0);
      }
    }
    TT_LOG_INFO("[SingleProcessWorker] Worker {} exited", worker_id);
  }
}

}  // namespace tt::worker
