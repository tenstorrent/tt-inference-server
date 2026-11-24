// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>

#include "config/runner_config.hpp"
#include "domain/video_generate_request.hpp"
#include "runtime/runners/media_runner.hpp"

namespace tt::runners {

/** Device-free video runner. Returns a `mock://video/<task_id>` reference
 *  so the HTTP and IPC path can be exercised before a DiT runner exists. */
class MockVideoRunner
    : public IMediaRunner<tt::domain::VideoGenerateRequest, std::string> {
 public:
  explicit MockVideoRunner(config::VideoConfig config);

  const char* runnerType() const override { return "MockVideoRunner"; }
  std::string run(const tt::domain::VideoGenerateRequest& request) override;
  /** Test hook: TT_VIDEO_MOCK_WARMUP_MS keeps the service "not ready". */
  bool warmup() override;

 private:
  config::VideoConfig videoConfig;
};

}  // namespace tt::runners
