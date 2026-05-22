// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/image_ipc_runner.hpp"

#include <json/json.h>

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

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
    : SyncMediaIpcRunner("ImageIpcRunner", workerId),
      config_(std::move(config)) {}

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
  TT_LOG_INFO("[ImageIpcRunner] Worker {} warming media runner ({})", workerId(),
              runner_->runnerType());
  return runner_->warmup();
}

void ImageIpcRunner::stop() {
  if (runner_) {
    runner_->stop();
  }
  SyncMediaIpcRunner::stop();
}

void ImageIpcRunner::processTask(
    const tt::ipc::file_payload::FilePayloadTask& task,
    tt::ipc::file_payload::FilePayloadResult& result) {
  const auto started = std::chrono::steady_clock::now();
  Json::Value requestJson = readJsonFile(task.request_path);
  auto request =
      tt::domain::ImageGenerateRequest::fromJson(requestJson, task.task_id);

  tt::domain::image::ImageResponse response(task.task_id);
  response.images = runner_->run(request);
  response.generation_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  writeResponseFile(task.response_path, response);
  result.generation_time_seconds = response.generation_time_seconds;
}

}  // namespace tt::runners
