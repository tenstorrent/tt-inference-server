// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "utils/tokenizer.hpp"

namespace tt::utils {

class DeepseekTokenizer final : public Tokenizer {
public:
    using Tokenizer::Tokenizer;

    std::string model_name() const override { return "deepseek-ai/DeepSeek-R1-0528"; }
    int special_token_decode_threshold() const override { return -1; }
    std::vector<int64_t> stop_token_ids() const override { return {1}; }

    std::string apply_chat_template(
        const std::vector<domain::ChatMessage>& messages,
        bool add_generation_prompt) const override;
};

}  // namespace tt::utils
