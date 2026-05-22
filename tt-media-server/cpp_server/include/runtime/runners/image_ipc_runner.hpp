// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
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
  Json::Value processJsonTask(const Json::Value& request,
                              uint32_t taskId) override;

  config::ImageConfig config_;
  std::unique_ptr<MediaRunner> runner_;
};

}  // namespace tt::runners
