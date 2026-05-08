// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/image_controller.hpp"

#include "api/sync_media_dispatch.hpp"
#include "config/settings.hpp"
#include "domain/image_generate_request.hpp"
#include "services/service_container.hpp"
#include "utils/logger.hpp"

namespace tt::api {

ImageController::ImageController() {
  if (!tt::config::isImageService()) {
    return;
  }
  service_ = std::dynamic_pointer_cast<tt::services::ImageService>(
      tt::services::ServiceContainer::instance().getService(
          tt::config::ModelService::IMAGE));
  if (!service_) {
    throw std::runtime_error(
        "[ImageController] Image service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[ImageController] Initialized");
}

ImageController::~ImageController() = default;

void ImageController::generate(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  dispatchJsonRequest<tt::services::ImageService,
                      tt::domain::ImageGenerateRequest>(
      req, std::move(callback), service_, request_counter_, "ImageController",
      "generations", {"prompt"});
}

void ImageController::imageToImage(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  dispatchJsonRequest<tt::services::ImageService,
                      tt::domain::ImageGenerateRequest>(
      req, std::move(callback), service_, request_counter_, "ImageController",
      "image-to-image", {"prompt", "image"});
}

void ImageController::edit(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  dispatchJsonRequest<tt::services::ImageService,
                      tt::domain::ImageGenerateRequest>(
      req, std::move(callback), service_, request_counter_, "ImageController",
      "edits", {"prompt", "image", "mask"});
}

}  // namespace tt::api
