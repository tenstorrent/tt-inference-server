// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer_strategy.hpp"
#include "utils/tokenizer.hpp"
#include "config/settings.hpp"

#include <iostream>
#include <sstream>

namespace tt::utils {

// ---------------------------------------------------------------------------
// DeepSeek V3: full-width vertical line U+FF5C (｜) delimiters
// ---------------------------------------------------------------------------

static const char* DS_USER_TAG = "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>";
static const char* DS_ASSISTANT_TAG = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";

std::string DeepSeekTokenizerStrategy::apply_chat_template(
    const std::vector<domain::ChatMessage>& messages,
    bool add_generation_prompt,
    const TokenizerConfig& cfg) const {

    std::ostringstream out;

    if (cfg.add_bos_token) out << cfg.bos_token;

    for (const auto& m : messages) {
        if (m.role == "system") out << m.content;
    }

    for (const auto& m : messages) {
        if (m.role == "system") continue;
        if (m.role == "user") {
            out << DS_USER_TAG << m.content;
        } else if (m.role == "assistant") {
            out << DS_ASSISTANT_TAG << m.content;
            if (cfg.add_eos_token) out << cfg.eos_token;
        }
    }

    if (add_generation_prompt) {
        out << DS_ASSISTANT_TAG;
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// Llama 3.1 8B: header/eot tags with knowledge-cutoff preamble
// ---------------------------------------------------------------------------

static const char* LLAMA_HEADER_START = "<|start_header_id|>";
static const char* LLAMA_HEADER_END = "<|end_header_id|>";
static const char* LLAMA_EOT = "<|eot_id|>";
static const char* LLAMA_SYSTEM_PREAMBLE =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

std::string LlamaTokenizerStrategy::apply_chat_template(
    const std::vector<domain::ChatMessage>& messages,
    bool add_generation_prompt,
    const TokenizerConfig& cfg) const {

    std::ostringstream out;

    std::string system_content;
    for (const auto& m : messages) {
        if (m.role == "system") {
            if (!system_content.empty()) system_content += "\n\n";
            system_content += m.content;
        }
    }

    if (cfg.add_bos_token) out << cfg.bos_token;

    out << LLAMA_HEADER_START << "system" << LLAMA_HEADER_END << "\n\n"
        << LLAMA_SYSTEM_PREAMBLE
        << system_content
        << LLAMA_EOT;

    for (const auto& m : messages) {
        if (m.role == "system") continue;
        std::string role = m.role.empty() ? "user" : m.role;
        out << LLAMA_HEADER_START << role << LLAMA_HEADER_END << "\n\n"
            << m.content << LLAMA_EOT;
    }

    if (add_generation_prompt) {
        out << LLAMA_HEADER_START << "assistant" << LLAMA_HEADER_END << "\n\n";
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// Factory + global singleton
// ---------------------------------------------------------------------------

std::unique_ptr<ITokenizerStrategy> create_tokenizer_strategy(config::RunnerType runner_type) {
    switch (runner_type) {
        case config::RunnerType::LLAMA_RUNNER:
            return std::make_unique<LlamaTokenizerStrategy>();
        case config::RunnerType::LLM_TEST:
        default:
            return std::make_unique<DeepSeekTokenizerStrategy>();
    }
}

const ITokenizerStrategy& active_tokenizer_strategy() {
    static auto strategy = create_tokenizer_strategy(config::model_runner_type());
    return *strategy;
}

}  // namespace tt::utils
