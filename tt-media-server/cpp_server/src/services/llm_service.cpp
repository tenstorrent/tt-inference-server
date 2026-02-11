// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "services/llm_service.hpp"
#include "config/settings.hpp"

#include <iostream>
#include <variant>

namespace tt::services {

LLMService::LLMService() : BaseService() {
    std::string path = tt::config::tokenizer_path();
    if (!path.empty()) {
        tokenizer_ = tt::utils::TokenizerUtil::load(path);
        if (tokenizer_.is_loaded()) {
            std::cout << "[LLMService] Tokenizer loaded from " << path << std::endl;
        }
    }
}

domain::CompletionRequest LLMService::pre_process(domain::CompletionRequest request) {
    if (!tokenizer_.is_loaded()) {
        return request;
    }
    if (std::holds_alternative<std::string>(request.prompt)) {
        const std::string& text = std::get<std::string>(request.prompt);
        std::vector<int> ids = tokenizer_.encode(text);
        request.prompt = std::move(ids);
    }
    return request;
}

}  // namespace tt::services
