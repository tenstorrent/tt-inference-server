// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace tt::config {

enum class ModelType {
    LLAMA_3_1_8B,
    DEEPSEEK_V3,
};

#ifdef MODEL_DEEPSEEK_V3

inline constexpr ModelType ACTIVE_MODEL = ModelType::DEEPSEEK_V3;
inline constexpr std::string_view MODEL_NAME = "deepseek-ai/DeepSeek-V3";

// DeepSeek V3 does not use a contiguous special-token ID range; no decode filtering needed.
inline constexpr int SPECIAL_TOKEN_DECODE_THRESHOLD = -1;

inline std::vector<int64_t> default_stop_token_ids() { return {1}; }

#else  // Default: meta-llama/Llama-3.1-8B

inline constexpr ModelType ACTIVE_MODEL = ModelType::LLAMA_3_1_8B;
inline constexpr std::string_view MODEL_NAME = "meta-llama/Llama-3.1-8B";

// Llama 3 special tokens occupy IDs [128000, 128255]; suppress in decoded output.
inline constexpr int SPECIAL_TOKEN_DECODE_THRESHOLD = 128000;

inline std::vector<int64_t> default_stop_token_ids() { return {128001, 128008, 128009}; }

#endif

}  // namespace tt::config
