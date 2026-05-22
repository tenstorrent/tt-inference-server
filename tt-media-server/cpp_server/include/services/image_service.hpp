// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image/image_response.hpp"
#include "domain/image_generate_request.hpp"
#include "ipc/image_ipc.hpp"
#include "runtime/runners/media_runner.hpp"
#include "runtime/worker/worker_manager.hpp"
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
  ImageService(config::ImageConfig config,
               std::unique_ptr<tt::worker::WorkerManager> workerManager,
               std::unique_ptr<tt::ipc::image::ImageQueueManager> queueManager);
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
  void startWorkerConsumers();
  void consumerLoopForWorker(size_t workerIdx);
  void stopWorkerMode();
  domain::image::ImageResponse processInProcessRequest(
      const domain::ImageGenerateRequest& request);
  domain::image::ImageResponse processWorkerRequest(
      const domain::ImageGenerateRequest& request);

  config::ImageConfig config_;
  RunnerList runners_;
  std::unique_ptr<tt::worker::WorkerManager> worker_manager_;
  std::unique_ptr<tt::ipc::image::ImageQueueManager> image_queue_manager_;
  std::vector<std::thread> consumer_threads_;
  mutable std::mutex pending_mutex_;
  std::unordered_map<uint32_t,
                     std::shared_ptr<std::promise<tt::ipc::image::ImageResult>>>
      pending_results_;
  std::filesystem::path ipc_payload_dir_;
  mutable std::atomic<size_t> next_runner_{0};
  std::vector<std::atomic<size_t>> runner_in_flight_;
  std::atomic<bool> ready_{false};
  std::atomic<bool> running_{false};
  mutable std::atomic<size_t> in_flight_{0};
  // Warmup runs here so start() can return immediately and the HTTP listener
  // can bind; /tt-liveness reports model_ready=false until warmup completes.
  std::thread warmup_thread_;
  std::mutex warmup_mutex_;
};

}  // namespace tt::services
