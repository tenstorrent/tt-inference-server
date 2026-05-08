// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "domain/image_response.hpp"
#include "runners/media_runner.hpp"
#include "services/base_service.hpp"

namespace tt::services {

/**
 * In-process service for image generation / image-to-image / edit endpoints.
 * Owns one runner and dispatches requests synchronously; the Drogon
 * controller offloads to a thread pool so the I/O loop is never blocked.
 */
class ImageService
    : public BaseService<domain::ImageGenerateRequest, domain::ImageResponse> {
 public:
  explicit ImageService(config::ImageConfig config);
  ~ImageService() override;

  ImageService(const ImageService&) = delete;
  ImageService& operator=(const ImageService&) = delete;

  void start() override;
  void stop() override;
  bool isModelReady() const override;

 protected:
  domain::ImageResponse processRequest(
      domain::ImageGenerateRequest request) override;

 private:
  config::ImageConfig config_;
  std::unique_ptr<runners::MediaRunner<domain::ImageGenerateRequest,
                                       std::vector<std::string>>>
      runner_;
  std::atomic<bool> ready_{false};
};

}  // namespace tt::services
