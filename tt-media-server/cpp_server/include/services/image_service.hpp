// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image/image_response.hpp"
#include "domain/image_generate_request.hpp"
#include "ipc/media_payload_ipc.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/base_service.hpp"
#include "services/media_worker_scheduler.hpp"

namespace tt::services {

/** Image service facade backed by media worker processes. */
class ImageService : public BaseSyncService<domain::ImageGenerateRequest,
                                            domain::image::ImageResponse> {
 public:
  ImageService(config::ImageConfig config,
               std::unique_ptr<tt::worker::WorkerManager> workerManager,
               std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet>
                   queueManager);
  ~ImageService() override;

  ImageService(const ImageService&) = delete;
  ImageService& operator=(const ImageService&) = delete;

  void start() override;
  void stop() override;
  bool isModelReady() const override;
  std::string runnerInUse() const override;

 protected:
  domain::image::ImageResponse produceResponse(
      domain::ImageGenerateRequest request) override;
  void preProcess(domain::ImageGenerateRequest& request) const override;
  size_t currentQueueSize() const override;
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

 private:
  config::ImageConfig imageConfig;
  std::unique_ptr<MediaWorkerScheduler> workerScheduler;
  mutable std::atomic<size_t> inFlight{0};
};

}  // namespace tt::services
