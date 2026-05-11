// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image/image_response.hpp"
#include "domain/image_generate_request.hpp"
#include "runners/media_runner.hpp"
#include "services/base_service.hpp"

namespace tt::services {

/** In-process image service. Owns one runner and dispatches synchronously;
 *  the Drogon controller offloads to a thread pool. */
class ImageService : public BaseService<domain::ImageGenerateRequest,
                                        domain::image::ImageResponse> {
 public:
  using Runner = runners::IMediaRunner<domain::ImageGenerateRequest,
                                       std::vector<std::string>>;

  ImageService(config::ImageConfig config, std::unique_ptr<Runner> runner);
  ~ImageService() override;

  ImageService(const ImageService&) = delete;
  ImageService& operator=(const ImageService&) = delete;

  void start() override;
  void stop() override;
  bool isModelReady() const override;

 protected:
  domain::image::ImageResponse processRequest(
      domain::ImageGenerateRequest request) override;
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

 private:
  config::ImageConfig config_;
  std::unique_ptr<Runner> runner_;
  std::atomic<bool> ready_{false};
  // Warmup (Python init + model load) runs on this thread so start() can
  // return immediately and the HTTP listener can bind; /tt-liveness reports
  // model_ready=false until warmup completes.
  std::thread warmup_thread_;
  std::mutex warmup_mutex_;
};

}  // namespace tt::services
