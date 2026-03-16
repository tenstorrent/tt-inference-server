// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <variant>

#include "config/llm_config.hpp"
#include "config/embedding_config.hpp"

namespace tt::config {

/**
 * Variant wrapper for all runner configuration types.
 * Allows generic handling of different service configurations.
 */
using RunnerConfig = std::variant<LLMConfig, EmbeddingConfig>;

}  // namespace tt::config
