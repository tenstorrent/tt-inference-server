// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/media_ipc_runner.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
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
    : runnerName(std::move(runnerName)), workerIndex(workerId) {
  taskQueue = std::make_unique<tt::ipc::media_payload::MediaPayloadTaskQueue>(
      tt::config::ttMediaTaskQueueName());
  resultQueue =
      std::make_unique<tt::ipc::media_payload::MediaPayloadResultQueue>(
          std::string(tt::config::ttMediaResultQueueName()) +
          std::to_string(workerIndex));
}

MediaIpcRunner::~MediaIpcRunner() { stop(); }

void MediaIpcRunner::stop() { stopped.store(true, std::memory_order_release); }

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
  TT_LOG_INFO("[MediaIpcRunner] Worker {} entering {} request loop",
              workerIndex, runnerName);
  while (!stopped.load(std::memory_order_acquire)) {
    // Bounded wait so a SIGTERM (e.g. PR_SET_PDEATHSIG after the parent
    // died) is noticed via `stopped` even when no task ever arrives.
    tt::ipc::media_payload::MediaPayloadTask task;
    if (!taskQueue->timedReceive(task,
                                 tt::config::defaults::WORKER_TASK_POLL_MS)) {
      continue;
    }
    if (task.isDone()) {
      TT_LOG_INFO("[MediaIpcRunner] Worker {} received shutdown task",
                  workerIndex);
      break;
    }

    // A cancel is a marker file next to the request (see VideoService::
    // cancelJob). It is checked twice: right after the pop, and again after
    // CLAIMED was sent. A cancel that lands between the two (before the service
    // saw CLAIMED) would otherwise still send the task to the device.
    auto cancelledBeforeStart = [&]() {
      const auto marker = tt::ipc::media_payload::cancelMarkerPath(
          task.request_path, task.task_id);
      return !marker.empty() && std::filesystem::exists(marker);
    };
    // Report it without running the model, so the device stays free for the
    // next job. The service finishes the job as cancelled without an error.
    auto skipCancelled = [&](const char* when) {
      tt::ipc::media_payload::MediaPayloadResult skipped;
      skipped.task_id = task.task_id;
      skipped.response_path = task.response_path;
      skipped.error = tt::ipc::media_payload::kCancelledBeforeStartError;
      TT_LOG_INFO("[MediaIpcRunner] Worker {} skipping cancelled task {} ({})",
                  workerIndex, task.task_id, when);
      resultQueue->push(skipped);
      std::error_code removeEc;
      std::filesystem::remove(task.request_path, removeEc);
    };

    if (cancelledBeforeStart()) {
      skipCancelled("before claim");
      continue;
    }

    // Test hook: widens the window between the first check and CLAIMED so a
    // cancel that lands in it can be reproduced (server_cases.py, phase
    // cancel-race). Not set in production.
    static const long preClaimDelayMs = [] {
      const char* v = std::getenv("TT_MEDIA_TEST_PRECLAIM_DELAY_MS");
      return (v && *v) ? std::strtol(v, nullptr, 10) : 0L;
    }();
    if (preClaimDelayMs > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(preClaimDelayMs));
    }

    resultQueue->push(
        tt::ipc::media_payload::MediaPayloadResult::claimed(task.task_id));

    if (cancelledBeforeStart()) {
      skipCancelled("after claim");
      continue;
    }

    tt::ipc::media_payload::MediaPayloadResult result;
    result.task_id = task.task_id;
    result.response_path = task.response_path;
    try {
      processTask(task, result);
    } catch (const std::exception& e) {
      result.error = e.what();
      TT_LOG_ERROR("[MediaIpcRunner] Worker {} task {} failed: {}", workerIndex,
                   task.task_id, e.what());
    } catch (...) {
      result.error = "unknown media runner error";
      TT_LOG_ERROR("[MediaIpcRunner] Worker {} task {} failed: unknown",
                   workerIndex, task.task_id);
    }

    if (!resultQueue->push(result)) {
      TT_LOG_ERROR(
          "[MediaIpcRunner] Worker {} failed to push result for task {}",
          workerIndex, task.task_id);
    }

    std::error_code ec;
    std::filesystem::remove(task.request_path, ec);
  }
}

}  // namespace tt::runners
