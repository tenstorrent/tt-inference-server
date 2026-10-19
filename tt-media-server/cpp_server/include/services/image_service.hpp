// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "config/runner_config.hpp"
#include "domain/image/image_response.hpp"
#include "domain/image_generate_request.hpp"
#include "ipc/media_payload_ipc.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/request_pipeline.hpp"

namespace tt::services {

/** Image service backed by WorkerManager + MediaPayloadQueueSet. */
class ImageService
    : public BaseAsyncService<domain::ImageGenerateRequest,
                              domain::image::ImageResponse,
                              tt::ipc::media_payload::MediaPayloadQueueSet> {
 public:
  ImageService(config::ImageConfig config,
               std::unique_ptr<tt::worker::WorkerManager> workerManager,
               std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet>
                   queueManager);
  ~ImageService() override;

  std::string runnerInUse() const override;

 protected:
  std::string validateRequest(
      const domain::ImageGenerateRequest& request) const override;
  IpcTask buildIpcTask(const domain::ImageGenerateRequest& request) override;
  domain::image::ImageResponse buildResponse(uint32_t taskId,
                                             const IpcResult& result) override;
  void cleanupTask(uint32_t taskId) override;
  void onStopped() override;

 private:
  std::string payloadPath(uint32_t taskId, const char* prefix) const;

  config::ImageConfig imageConfig_;
  std::filesystem::path payloadDir_;
};

}  // namespace tt::services
