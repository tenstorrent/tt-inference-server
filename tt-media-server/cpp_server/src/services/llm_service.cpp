// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/llm_service.hpp"
#include "config/settings.hpp"

#include <iostream>
#include <variant>

namespace tt::services {

LLMService::LLMService() : BaseService() {
    std::string path = tt::config::tokenizer_path();
    tokenizer_ = tt::utils::TokenizerUtil(path);
    std::cout << "[LLMService] Tokenizer path: " << (path.empty() ? "(none)" : path) << std::endl;
}

domain::CompletionRequest LLMService::pre_process(domain::CompletionRequest request) {
    if (std::holds_alternative<std::string>(request.prompt)) {
        const std::string& text = std::get<std::string>(request.prompt);
        std::vector<int> ids = tokenizer_.encode(text);
        if (!ids.empty()) {
            request.prompt = std::move(ids);
        }
    }
    return request;
}

}  // namespace tt::services
