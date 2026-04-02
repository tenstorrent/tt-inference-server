// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/mock_sp_pipeline_model_runner.hpp"

#include "profiling/tracy.hpp"

namespace sp_pipeline {

MockSpPipelineModelRunner::MockSpPipelineModelRunner(DecodeCallback callback,
                                                     MockDeviceConfig config)
    : decode_callback_(std::move(callback)), device_(config) {
  reader_thread_ = std::thread([this] { reader_loop(); });
}

MockSpPipelineModelRunner::~MockSpPipelineModelRunner() { exit(); }

void MockSpPipelineModelRunner::write(uint32_t taskId,
                                      const std::vector<int64_t>& tokenIds,
                                      uint32_t maxTokens, RequestPhase phase) {
  ZoneScopedN("MockModelRunner::write");
  device_.write(taskId, tokenIds, maxTokens, phase);
}

void MockSpPipelineModelRunner::exit() {
  if (stop_.exchange(true)) return;
  device_.exit();
  if (reader_thread_.joinable()) reader_thread_.join();
}

void MockSpPipelineModelRunner::reader_loop() {
  while (!stop_.load(std::memory_order_relaxed)) {
    auto result = device_.read();
    if (!result) break;
    {
      ZoneScopedN("MockModelRunner::decode_callback");
      decode_callback_(*result);
    }
  }
}

}  // namespace sp_pipeline
