// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/media_ipc_runner.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

namespace {

Json::Value readJsonFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open media IPC request file: " + path);
  }
  Json::CharReaderBuilder builder;
  Json::Value json;
  std::string errors;
  if (!Json::parseFromStream(builder, input, &json, &errors)) {
    throw std::runtime_error("failed to parse media IPC request file: " +
                             errors);
  }
  return json;
}

void writeJsonFile(const std::string& path, const Json::Value& json) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open media IPC response file: " + path);
  }
  output << Json::writeString(builder, json);
}

}  // namespace

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

void MediaIpcRunner::processTask(
    const tt::ipc::media_payload::MediaPayloadTask& task,
    tt::ipc::media_payload::MediaPayloadResult& result) {
  const auto started = std::chrono::steady_clock::now();
  Json::Value responseJson =
      processJsonTask(readJsonFile(task.request_path), task.task_id);
  result.generation_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (!responseJson.isMember("generation_time")) {
    responseJson["generation_time"] =
        std::round(result.generation_time_seconds * 100.0) / 100.0;
  }
  writeJsonFile(task.response_path, responseJson);
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
