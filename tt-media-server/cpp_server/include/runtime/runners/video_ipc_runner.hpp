// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "domain/video_generate_request.hpp"
#include "runtime/runners/media_ipc_runner.hpp"
#include "runtime/runners/media_runner.hpp"

namespace tt::runners {

class VideoIpcRunner : public MediaIpcRunner {
 public:
  using MediaRunner =
      IMediaRunner<tt::domain::VideoGenerateRequest, std::string>;

  VideoIpcRunner(config::VideoConfig config, int workerId);
  ~VideoIpcRunner() override;

  bool warmup() override;
  void stop() override;
  const char* runnerType() const override { return "VideoIpcRunner"; }

 private:
  Json::Value processJsonTask(const Json::Value& request,
                              uint32_t taskId) override;

  config::VideoConfig videoConfig;
  std::unique_ptr<MediaRunner> runner;
};

}  // namespace tt::runners
