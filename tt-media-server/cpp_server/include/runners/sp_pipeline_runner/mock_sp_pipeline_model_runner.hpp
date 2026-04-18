// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "runners/sp_pipeline_runner/i_sp_pipeline_model_runner.hpp"
#include "runners/sp_pipeline_runner/mock_device_pipeline.hpp"

namespace tt::runners::sp_pipeline {

/// Drop-in replacement for SpPipelineModelRunner that uses a simulated
/// device pipeline instead of shared memory + an external Python process.
class MockSpPipelineModelRunner : public ISpPipelineModelRunner {
 public:
  explicit MockSpPipelineModelRunner(DecodeCallback callback,
                                     MockDeviceConfig config = {});
  ~MockSpPipelineModelRunner() override;

  MockSpPipelineModelRunner(const MockSpPipelineModelRunner&) = delete;
  MockSpPipelineModelRunner& operator=(const MockSpPipelineModelRunner&) =
      delete;

  void write(uint32_t task_id, const std::vector<int64_t>& token_ids,
             uint32_t max_tokens, RequestPhase phase, bool fastMode) override;
  void exit() override;

 private:
  void reader_loop();

  DecodeCallback decode_callback_;
  MockDevicePipeline device_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;
};

}  // namespace tt::runners::sp_pipeline
