// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "config/runner_config.hpp"
#include "domain/video_generate_request.hpp"
#include "runtime/runners/media_runner.hpp"

namespace tt::runners {

/** Device video runner. Warmup calls the Python DiT runner selected by
 *  `VideoConfig::pythonRunner`: `set_device()` opens the mesh, `warmup()`
 *  loads weights. `run()` returns the mp4 path written for that request. */
class VideoDeviceRunner
    : public IMediaRunner<tt::domain::VideoGenerateRequest, std::string> {
 public:
  explicit VideoDeviceRunner(config::VideoConfig config);
  ~VideoDeviceRunner() override;

  VideoDeviceRunner(const VideoDeviceRunner&) = delete;
  VideoDeviceRunner& operator=(const VideoDeviceRunner&) = delete;

  bool warmup() override;
  std::string run(const tt::domain::VideoGenerateRequest& request) override;
  void stop() override;
  const char* runnerType() const override { return "VideoDeviceRunner"; }

 private:
  struct PythonState;

  config::VideoConfig videoConfig;
  std::unique_ptr<PythonState> python;
  bool ready = false;
};

}  // namespace tt::runners
