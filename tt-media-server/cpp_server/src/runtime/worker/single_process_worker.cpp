// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "runtime/worker/single_process_worker.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <thread>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "ipc/boost/boost_warmup_signal_queue.hpp"
#include "profiling/tracy.hpp"
#include "utils/crash_handler.hpp"
#include "utils/ipc_runner_factory.hpp"
#include "utils/logger.hpp"

namespace tt::worker {

SingleProcessWorker::SingleProcessWorker(WorkerConfig& cfg)
    : cfg(std::move(cfg)) {
  pid = getpid();
  worker_id = cfg.worker_id;
  is_ready = true;
}

SingleProcessWorker::~SingleProcessWorker() = default;

void SingleProcessWorker::start() {
  // The exec'd worker entrypoint (main.cpp startWorker) already installed the
  // shared crash handlers before anything else ran; re-install here (same
  // handlers, same tag) so in-process uses of SingleProcessWorker (unit and
  // integration tests that never go through main) are covered too.
  tt::utils::installCrashHandlers(tt::config::logInstanceTag(worker_id));

  tracy_config::tracySetThreadName(
      ("Worker-" + std::to_string(cfg.worker_id)).c_str());

  for (const auto& [key, value] : cfg.env_vars) {
    setenv(key.c_str(), value.c_str(), 1);
  }

  {
    ZoneScopedN("Worker::init");
    if (tt::config::isTtsService()) {
      auto* taskQueue =
          std::get_if<std::shared_ptr<tt::ipc::tts::TtsTaskQueue>>(
              &cfg.task_queue);
      auto* audioQueue =
          std::get_if<std::shared_ptr<tt::ipc::tts::TtsAudioChunkQueue>>(
              &cfg.result_queue);
      if (!taskQueue || !audioQueue || !*taskQueue || !*audioQueue ||
          !cfg.cancel_queue) {
        throw std::runtime_error(
            "TTS worker requires TTS task/audio queues and a cancel queue");
      }
      runner_ = tt::utils::ipc_runner_factory::createTtsIpcRunner(
          cfg.runner_config, taskQueue->get(), audioQueue->get(),
          cfg.cancel_queue.get());
    } else if (tt::config::isImageService()) {
      runner_ = tt::utils::ipc_runner_factory::createIpcRunner(
          tt::config::ModelService::IMAGE, cfg.runner_config, nullptr, nullptr,
          nullptr);
    } else {
      auto* taskQueue =
          std::get_if<std::shared_ptr<tt::ipc::ITaskQueue>>(&cfg.task_queue);
      auto* resultQueue = std::get_if<std::shared_ptr<tt::ipc::IResultQueue>>(
          &cfg.result_queue);
      if (!taskQueue || !resultQueue || !*taskQueue || !*resultQueue ||
          !cfg.cancel_queue) {
        throw std::runtime_error(
            "LLM worker requires LLM task/result queues and a cancel queue");
      }
      runner_ = tt::utils::ipc_runner_factory::createIpcRunner(
          tt::config::modelService(), cfg.runner_config, resultQueue->get(),
          taskQueue->get(), cfg.cancel_queue.get());
    }
  }
  TT_LOG_INFO(
      "[SingleProcessWorker] Worker {} starting runner (warmup may take a "
      "while)",
      worker_id);
  try {
    runner_->start([this]() {
      try {
        tt::ipc::boost::WarmupSignalQueue warmupQueue(
            tt::config::ttWarmupSignalsQueueName());
        warmupQueue.sendReady(worker_id);
        TT_LOG_INFO("[SingleProcessWorker] Worker {} signaled warmup complete",
                    worker_id);
      } catch (const std::exception& e) {
        TT_LOG_ERROR(
            "[SingleProcessWorker] Worker {} failed to signal warmup: {}",
            worker_id, e.what());
      }
    });
    TT_LOG_CRITICAL(
        "[SingleProcessWorker] Worker {} runner loop returned unexpectedly "
        "(runner type: {})",
        worker_id, runner_ ? runner_->runnerType() : "unknown");
  } catch (const std::exception& e) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorker] Worker {} CRASHED with exception: {} "
        "(runner type: {})",
        worker_id, e.what(), runner_ ? runner_->runnerType() : "unknown");
    throw;
  } catch (...) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorker] Worker {} CRASHED with unknown exception "
        "(runner type: {})",
        worker_id, runner_ ? runner_->runnerType() : "unknown");
    throw;
  }
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
