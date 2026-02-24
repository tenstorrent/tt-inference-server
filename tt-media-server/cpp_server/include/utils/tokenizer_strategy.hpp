// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config/constants.hpp"
#include "domain/chat_message.hpp"

namespace tt::utils {

struct TokenizerConfig;

/**
 * Strategy interface for model-specific tokenizer behavior.
 * Implementations encapsulate chat template formatting, special token handling,
 * and model metadata. Selected at runtime based on RunnerType.
 */
class ITokenizerStrategy {
public:
    virtual ~ITokenizerStrategy() = default;
    virtual std::string model_name() const = 0;
    virtual int special_token_decode_threshold() const = 0;
    virtual std::vector<int64_t> stop_token_ids() const = 0;
    virtual std::string tokenizer_dir_name() const = 0;
    virtual std::string apply_chat_template(
        const std::vector<domain::ChatMessage>& messages,
        bool add_generation_prompt,
        const TokenizerConfig& config) const = 0;
};

class DeepSeekTokenizerStrategy final : public ITokenizerStrategy {
public:
    std::string model_name() const override { return "deepseek-ai/DeepSeek-V3"; }
    int special_token_decode_threshold() const override { return -1; }
    std::vector<int64_t> stop_token_ids() const override { return {1}; }
    std::string tokenizer_dir_name() const override { return "deepseek-ai/DeepSeek-V3"; }
    std::string apply_chat_template(
        const std::vector<domain::ChatMessage>& messages,
        bool add_generation_prompt,
        const TokenizerConfig& config) const override;
};

class LlamaTokenizerStrategy final : public ITokenizerStrategy {
public:
    std::string model_name() const override { return "meta-llama/Llama-3.1-8B"; }
    int special_token_decode_threshold() const override { return 128000; }
    std::vector<int64_t> stop_token_ids() const override { return {128001, 128008, 128009}; }
    std::string tokenizer_dir_name() const override { return "meta-llama/Llama-3.1-8B"; }
    std::string apply_chat_template(
        const std::vector<domain::ChatMessage>& messages,
        bool add_generation_prompt,
        const TokenizerConfig& config) const override;
};

/**
 * Factory: creates tokenizer strategy based on runner type.
 * LLM_TEST -> DeepSeekTokenizerStrategy (default)
 * LLAMA_RUNNER -> LlamaTokenizerStrategy
 */
std::unique_ptr<ITokenizerStrategy> create_tokenizer_strategy(config::RunnerType runner_type);

/**
 * Global active tokenizer strategy, auto-initialized from MODEL_RUNNER env var on first access.
 * Thread-safe (C++11 function-local static initialization).
 */
const ITokenizerStrategy& active_tokenizer_strategy();

}  // namespace tt::utils
