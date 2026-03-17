// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/mock_sp_pipeline_model_runner.hpp"

#include "profiling/tracy.hpp"

namespace sp_pipeline {

MockSpPipelineModelRunner::MockSpPipelineModelRunner(
    DecodeCallback callback, MockDeviceConfig config)
    : decode_callback_(std::move(callback)), device_(config) {
  reader_thread_ = std::thread([this] {
    tracy_config::TracySetThreadName("MockDevice::reader");
    reader_loop();
  });
}

MockSpPipelineModelRunner::~MockSpPipelineModelRunner() { exit(); }

void MockSpPipelineModelRunner::write(const std::string& task_id,
                                      const std::vector<int64_t>& token_ids,
                                      uint32_t max_tokens,
                                      RequestPhase phase) {
  ZoneScopedN("MockModelRunner::write");
  device_.write(task_id, token_ids, max_tokens, phase);
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
