// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "config/runner_config.hpp"
#include "domain/video_generate_request.hpp"
#include "runtime/runners/media_runner.hpp"

namespace tt::runners {

/** Device video runner: creates the C++ runner (runtime/dit) for
 *  `VideoConfig::pythonRunner`. It opens the mesh, validates requests, builds
 *  the pipeline calls and exports the mp4; the only Python it runs is the
 *  model's adapter in `python/cpp_dit_adapters`, which calls tt_dit.
 *  `run()` returns the mp4 path written for that request. */
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

  /** Sets the process environment the runner needs (weight cache, visible
   *  devices, the model's flag defaults). Idempotent; call it before other
   *  threads exist when the runner runs inside the server. */
  static void prepareProcessEnvironment(const config::VideoConfig& config);

 private:
  struct PythonState;

  config::VideoConfig videoConfig;
  std::unique_ptr<PythonState> python;
  bool ready = false;
};

}  // namespace tt::runners
