// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/media_ipc_runner.hpp"

#include <filesystem>
#include <utility>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

MediaIpcRunner::MediaIpcRunner(std::string runnerName, int workerId)
    : runner_name_(std::move(runnerName)), worker_id_(workerId) {
  task_queue_ = std::make_unique<tt::ipc::media_payload::MediaPayloadTaskQueue>(
      tt::config::ttTaskQueueName());
  result_queue_ =
      std::make_unique<tt::ipc::media_payload::MediaPayloadResultQueue>(
          std::string(tt::config::ttResultQueueName()) +
          std::to_string(worker_id_));
}

MediaIpcRunner::~MediaIpcRunner() { stop(); }

void MediaIpcRunner::stop() {
  stopped_.store(true, std::memory_order_release);
}

void MediaIpcRunner::run() {
  TT_LOG_INFO("[MediaIpcRunner] Worker {} entering {} request loop", worker_id_,
              runner_name_);
  while (!stopped_.load(std::memory_order_acquire)) {
    tt::ipc::media_payload::MediaPayloadTask task;
    task_queue_->receive(task);
    if (task.isDone()) {
      TT_LOG_INFO("[MediaIpcRunner] Worker {} received shutdown task",
                  worker_id_);
      break;
    }

    tt::ipc::media_payload::MediaPayloadResult result;
    result.task_id = task.task_id;
    result.response_path = task.response_path;
    try {
      processTask(task, result);
    } catch (const std::exception& e) {
      result.error = e.what();
      TT_LOG_ERROR("[MediaIpcRunner] Worker {} task {} failed: {}", worker_id_,
                   task.task_id, e.what());
    } catch (...) {
      result.error = "unknown media runner error";
      TT_LOG_ERROR("[MediaIpcRunner] Worker {} task {} failed: unknown",
                   worker_id_, task.task_id);
    }

    if (!result_queue_->push(result)) {
      TT_LOG_ERROR("[MediaIpcRunner] Worker {} failed to push result for task {}",
                   worker_id_, task.task_id);
    }

    std::error_code ec;
    std::filesystem::remove(task.request_path, ec);
  }
}

}  // namespace tt::runners
