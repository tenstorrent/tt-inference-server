// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "services/base_service.hpp"
#include "utils/tokenizer.hpp"

namespace tt::services {

/**
 * LLM Service for text completions.
 * Tokenizes string prompts in pre_process (vLLM-style); workers detokenize when returning results.
 */
class LLMService : public BaseService {
public:
    LLMService();

protected:
    domain::CompletionRequest pre_process(domain::CompletionRequest request) override;

private:
    tt::utils::TokenizerUtil tokenizer_;
};

}  // namespace tt::services
