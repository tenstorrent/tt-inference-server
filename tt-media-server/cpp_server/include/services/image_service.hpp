// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "domain/image_response.hpp"
#include "runners/image_runner.hpp"
#include "services/base_service.hpp"

namespace tt::services {

/**
 * In-process service for image generation / image-to-image / edit endpoints.
 *
 * Image generation is batch-1 / seconds-long, so unlike LLM/Embedding the
 * service does not spawn worker subprocesses; it owns one ImageRunner and
 * dispatches requests to it directly. The Drogon controller hands off to a
 * background thread pool so the I/O loop is never blocked.
 *
 * P2+ subclasses or alternative service impls can swap in a worker-process
 * model if Python interpreter isolation becomes necessary for SDXL.
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
  void postProcess(domain::ImageResponse& /*response*/) const override {}
  size_t currentQueueSize() const override { return 0; }

 private:
  config::ImageConfig config_;
  std::unique_ptr<runners::ImageRunner> runner_;
  std::atomic<bool> ready_{false};
};

}  // namespace tt::services
