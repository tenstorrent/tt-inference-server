// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "utils/tokenizer.hpp"

namespace tt::utils {

class LlamaTokenizer final : public Tokenizer {
public:
    using Tokenizer::Tokenizer;

    std::string model_name() const override { return "meta-llama/Llama-3.1-8B-Instruct"; }
    int special_token_decode_threshold() const override { return 128000; }
    std::vector<int64_t> stop_token_ids() const override { return {128001, 128008, 128009}; }

    std::string apply_chat_template(
        const std::vector<domain::ChatMessage>& messages,
        bool add_generation_prompt) const override;
};

}  // namespace tt::utils
