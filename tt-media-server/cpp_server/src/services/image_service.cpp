// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/image_service.hpp"

#include <json/json.h>
#include <unistd.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "utils/logger.hpp"

namespace tt::services {

namespace {

void writeJsonFile(const std::filesystem::path& path, const Json::Value& json) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open media IPC payload file: " +
                             path.string());
  }
  output << Json::writeString(builder, json);
}

Json::Value readJsonFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open media IPC response file: " +
                             path.string());
  }
  Json::CharReaderBuilder builder;
  Json::Value json;
  std::string errors;
  if (!Json::parseFromStream(builder, input, &json, &errors)) {
    throw std::runtime_error("failed to parse media IPC response file: " +
                             errors);
  }
  return json;
}

domain::image::ImageResponse responseFromJson(uint32_t taskId,
                                              const Json::Value& json) {
  domain::image::ImageResponse response(taskId);
  if (json.isMember("images") && json["images"].isArray()) {
    for (const auto& image : json["images"]) {
      response.images.push_back(image.asString());
    }
  }
  if (json.isMember("generation_time")) {
    response.generation_time_seconds = json["generation_time"].asDouble();
  }
  if (json.isMember("error")) {
    response.error = json["error"].asString();
  }
  return response;
}

}  // namespace

ImageService::ImageService(
    config::ImageConfig config,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet> queueManager)
    : BaseAsyncService("ImageService", std::move(workerManager),
                       std::move(queueManager)),
      imageConfig_(std::move(config)),
      payloadDir_(std::filesystem::temp_directory_path() /
                  ("tt-image-ipc-" + std::to_string(::getpid()))) {
  std::filesystem::create_directories(payloadDir_);
  TT_LOG_INFO(
      "[ImageService] Initialized worker-backed image service "
      "(workers={}, max_queue_size={}, payload_dir='{}')",
      numWorkers(), this->maxQueueSize, payloadDir_.string());
}

ImageService::~ImageService() { stop(); }

std::string ImageService::runnerInUse() const {
  return config::toClientRunnerName(imageConfig_.runner_type);
}

std::string ImageService::validateRequest(
    const domain::ImageGenerateRequest& /*request*/) const {
  if (!isModelReady()) {
    return "Image service not ready";
  }
  return {};
}

ImageService::IpcTask ImageService::buildIpcTask(
    const domain::ImageGenerateRequest& request) {
  const auto reqPath = payloadPath(request.task_id, "request");
  const auto respPath = payloadPath(request.task_id, "response");
  writeJsonFile(reqPath, request.toJson());

  IpcTask task;
  task.task_id = request.task_id;
  task.request_path = reqPath;
  task.response_path = respPath;
  return task;
}

domain::image::ImageResponse ImageService::buildResponse(
    uint32_t taskId, const IpcResult& result) {
  domain::image::ImageResponse response(taskId);
  response.generation_time_seconds = result.generation_time_seconds;
  if (!result.error.empty()) {
    response.error = result.error;
    return response;
  }
  try {
    auto body = readJsonFile(payloadPath(taskId, "response"));
    response = responseFromJson(taskId, body);
    response.generation_time_seconds = result.generation_time_seconds;
  } catch (const std::exception& e) {
    response.error = e.what();
  }
  return response;
}

void ImageService::cleanupTask(uint32_t taskId) {
  std::error_code ec;
  std::filesystem::remove(payloadPath(taskId, "request"), ec);
  std::filesystem::remove(payloadPath(taskId, "response"), ec);
}

void ImageService::onStopped() {
  std::error_code ec;
  std::filesystem::remove_all(payloadDir_, ec);
}

std::string ImageService::payloadPath(uint32_t taskId,
                                      const char* prefix) const {
  return (payloadDir_ /
          (std::string(prefix) + "-" + std::to_string(taskId) + ".json"))
      .string();
}

}  // namespace tt::services
