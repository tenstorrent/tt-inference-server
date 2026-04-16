// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/i_sp_pipeline_model_runner.hpp"

#include <stdexcept>

#include "runners/sp_pipeline_runner/mock_sp_pipeline_model_runner.hpp"
#include "runners/sp_pipeline_runner/sp_pipeline_model_runner.hpp"

namespace tt::runners::sp_pipeline {

using ModelRunnerType = tt::config::ModelRunnerType;

std::unique_ptr<ISpPipelineModelRunner> makeModelRunner(
    const tt::config::LLMConfig& config, DecodeCallback callback) {
  switch (config.runner_type) {
    case ModelRunnerType::PIPELINE:
      return std::make_unique<SpPipelineModelRunner>(std::move(callback));
    case ModelRunnerType::MOCK_PIPELINE:
      return std::make_unique<MockSpPipelineModelRunner>(std::move(callback));
    default:
      throw std::invalid_argument(
          "Invalid model runner type for BlazeRunner");
  }
}

}  // namespace tt::runners::sp_pipeline
