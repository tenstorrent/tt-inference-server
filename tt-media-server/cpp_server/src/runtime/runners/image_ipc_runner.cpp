// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/image_ipc_runner.hpp"

#include <json/json.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "config/settings.hpp"
#include "domain/image/image_response.hpp"
#include "runtime/runners/runner_registry.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

namespace {

Json::Value readJsonFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open image IPC request file: " + path);
  }
  Json::CharReaderBuilder builder;
  Json::Value json;
  std::string errors;
  if (!Json::parseFromStream(builder, input, &json, &errors)) {
    throw std::runtime_error("failed to parse image IPC request file: " +
                             errors);
  }
  return json;
}

void writeResponseFile(const std::string& path,
                       const tt::domain::image::ImageResponse& response) {
  Json::Value json = response.toOpenaiJson();
  if (!response.error.empty()) {
    json["error"] = response.error;
  }
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open image IPC response file: " + path);
  }
  output << Json::writeString(builder, json);
}

}  // namespace

ImageIpcRunner::ImageIpcRunner(config::ImageConfig config, int workerId)
    : config_(std::move(config)), worker_id_(workerId) {
  task_queue_ = std::make_unique<tt::ipc::image::ImageTaskQueue>(
      tt::config::ttTaskQueueName());
  result_queue_ = std::make_unique<tt::ipc::image::ImageResultQueue>(
      std::string(tt::config::ttResultQueueName()) +
      std::to_string(worker_id_));
}

ImageIpcRunner::~ImageIpcRunner() { stop(); }

bool ImageIpcRunner::warmup() {
  runner_ = tt::utils::RunnerRegistry::instance().createMedia<MediaRunner>(
      config::ModelService::IMAGE, config_.runner_type,
      config::RunnerConfig{config_});
  if (!runner_) {
    throw std::runtime_error(
        "[ImageIpcRunner] no media runner registered for runner_type=" +
        config::toString(config_.runner_type));
  }
  TT_LOG_INFO("[ImageIpcRunner] Worker {} warming media runner ({})",
              worker_id_, runner_->runnerType());
  return runner_->warmup();
}

void ImageIpcRunner::stop() {
  stopped_.store(true, std::memory_order_release);
  if (runner_) {
    runner_->stop();
  }
}

void ImageIpcRunner::run() {
  TT_LOG_INFO("[ImageIpcRunner] Worker {} entering request loop", worker_id_);
  while (!stopped_.load(std::memory_order_acquire)) {
    tt::ipc::image::ImageTask task;
    task_queue_->receive(task);
    if (task.isDone()) {
      TT_LOG_INFO("[ImageIpcRunner] Worker {} received shutdown task",
                  worker_id_);
      break;
    }
    handleTask(task);
  }
}

void ImageIpcRunner::handleTask(const tt::ipc::image::ImageTask& task) {
  tt::ipc::image::ImageResult result;
  result.task_id = task.task_id;
  result.response_path = task.response_path;
  const auto started = std::chrono::steady_clock::now();
  try {
    Json::Value requestJson = readJsonFile(task.request_path);
    auto request =
        tt::domain::ImageGenerateRequest::fromJson(requestJson, task.task_id);

    tt::domain::image::ImageResponse response(task.task_id);
    response.images = runner_->run(request);
    response.generation_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    writeResponseFile(task.response_path, response);
    result.generation_time_seconds = response.generation_time_seconds;
  } catch (const std::exception& e) {
    result.error = e.what();
    TT_LOG_ERROR("[ImageIpcRunner] Worker {} task {} failed: {}", worker_id_,
                 task.task_id, e.what());
  }

  if (!result_queue_->push(result)) {
    TT_LOG_ERROR("[ImageIpcRunner] Worker {} failed to push result for task {}",
                 worker_id_, task.task_id);
  }

  std::error_code ec;
  std::filesystem::remove(task.request_path, ec);
}

}  // namespace tt::runners
