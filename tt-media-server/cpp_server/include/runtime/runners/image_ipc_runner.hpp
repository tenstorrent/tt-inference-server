// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "ipc/image_ipc.hpp"
#include "runtime/runners/ipc_runner.hpp"
#include "runtime/runners/media_runner.hpp"

namespace tt::runners {

class ImageIpcRunner : public IRunner {
 public:
  using MediaRunner =
      IMediaRunner<tt::domain::ImageGenerateRequest, std::vector<std::string>>;

  ImageIpcRunner(config::ImageConfig config, int workerId);
  ~ImageIpcRunner() override;

  bool warmup() override;
  void stop() override;
  const char* runnerType() const override { return "ImageIpcRunner"; }

 private:
  void run() override;
  void handleTask(const tt::ipc::image::ImageTask& task);

  config::ImageConfig config_;
  int worker_id_;
  std::unique_ptr<tt::ipc::image::ImageTaskQueue> task_queue_;
  std::unique_ptr<tt::ipc::image::ImageResultQueue> result_queue_;
  std::unique_ptr<MediaRunner> runner_;
  std::atomic<bool> stopped_{false};
};

}  // namespace tt::runners
