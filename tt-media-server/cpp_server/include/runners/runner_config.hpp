// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <variant>

#include "runners/llm_runner/config.hpp"
#include "runners/embedding_config.hpp"
#include "runners/sp_pipeline_runner/config.hpp"

namespace tt::runners {

using RunnerConfig = std::variant<llm_engine::Config, EmbeddingConfig, sp_pipeline::SpPipelineConfig>;

}  // namespace tt::runners
