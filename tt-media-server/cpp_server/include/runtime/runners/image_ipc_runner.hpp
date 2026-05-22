// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "ipc/media_payload_ipc.hpp"
#include "runtime/runners/media_ipc_runner.hpp"
#include "runtime/runners/media_runner.hpp"

namespace tt::runners {

class ImageIpcRunner : public MediaIpcRunner {
 public:
  using MediaRunner =
      IMediaRunner<tt::domain::ImageGenerateRequest, std::vector<std::string>>;

  ImageIpcRunner(config::ImageConfig config, int workerId);
  ~ImageIpcRunner() override;

  bool warmup() override;
  void stop() override;
  const char* runnerType() const override { return "ImageIpcRunner"; }

 private:
  void processTask(const tt::ipc::media_payload::MediaPayloadTask& task,
                   tt::ipc::media_payload::MediaPayloadResult& result) override;

  config::ImageConfig config_;
  std::unique_ptr<MediaRunner> runner_;
};

}  // namespace tt::runners
