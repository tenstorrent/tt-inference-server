// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/HttpController.h>

#include <atomic>
#include <memory>

#include "services/image_service.hpp"

namespace tt::api {

/**
 * OpenAI-compatible image generation controller. Only one image mode is
 * active per process: model_service_registration.cpp registers exactly the
 * route matching the configured runner_type, so the inactive endpoints are
 * not present in the RouteRegistry and 404 via SyncAdvice in main.cpp.
 */
class ImageController : public drogon::HttpController<ImageController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(ImageController::generate, "/v1/images/generations",
                drogon::Post);
  ADD_METHOD_TO(ImageController::imageToImage, "/v1/images/image-to-image",
                drogon::Post);
  ADD_METHOD_TO(ImageController::edit, "/v1/images/edits", drogon::Post);
  METHOD_LIST_END

  ImageController();
  ~ImageController();

  void generate(const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback);
  void imageToImage(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback);
  void edit(const drogon::HttpRequestPtr& req,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback);

 private:
  std::shared_ptr<services::ImageService> service_;
  std::atomic<uint64_t> request_counter_{0};
};

}  // namespace tt::api
