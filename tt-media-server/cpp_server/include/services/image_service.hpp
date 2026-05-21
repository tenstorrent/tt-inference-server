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
#include "runtime/runners/media_runner.hpp"
#include "services/base_service.hpp"

namespace tt::services {

/** In-process image service. Owns one or more runners and dispatches
 *  synchronously; the Drogon controller offloads to a thread pool. */
class ImageService : public BaseService<domain::ImageGenerateRequest,
                                        domain::image::ImageResponse> {
 public:
  using Runner = runners::IMediaRunner<domain::ImageGenerateRequest,
                                       std::vector<std::string>>;
  using RunnerList = std::vector<std::unique_ptr<Runner>>;

  ImageService(config::ImageConfig config, std::unique_ptr<Runner> runner);
  ImageService(config::ImageConfig config, RunnerList runners);
  ~ImageService() override;

  ImageService(const ImageService&) = delete;
  ImageService& operator=(const ImageService&) = delete;

  void start() override;
  void stop() override;
  bool isModelReady() const override;
  std::string runnerInUse() const override;

 protected:
  domain::image::ImageResponse processRequest(
      domain::ImageGenerateRequest request) override;
  void preProcess(domain::ImageGenerateRequest& request) const override;
  size_t currentQueueSize() const override;
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

 private:
  size_t selectRunnerIndex() const;

  config::ImageConfig config_;
  RunnerList runners_;
  mutable std::atomic<size_t> next_runner_{0};
  std::vector<std::atomic<size_t>> runner_in_flight_;
  std::atomic<bool> ready_{false};
  mutable std::atomic<size_t> in_flight_{0};
  // Warmup runs here so start() can return immediately and the HTTP listener
  // can bind; /tt-liveness reports model_ready=false until warmup completes.
  std::thread warmup_thread_;
  std::mutex warmup_mutex_;
};

}  // namespace tt::services
