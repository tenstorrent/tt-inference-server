// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "config/model_config.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace tt::utils {

Tokenizer::Tokenizer(const std::string& path) {
    if (path.empty()) {
        throw std::runtime_error("[TokenizerUtil] Cannot initialize with empty path");
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("[TokenizerUtil] Failed to open: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string blob = ss.str();
    f.close();

    if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".json") == 0) {
        tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    } else if (path.size() >= 7 && path.compare(path.size() - 7, 7, ".model") == 0) {
        tok_ = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
    } else {
        throw std::runtime_error("[TokenizerUtil] Unknown extension; use .json or .model: " + path);
    }

    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Failed to create tokenizer from: " + path);
    }

    std::cout << "[TokenizerUtil] Loaded tokenizer from: " << path
              << " (model: " << tt::config::MODEL_NAME << ")" << std::endl;
}

bool Tokenizer::is_loaded() const {
    return tok_ != nullptr;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer not loaded, cannot encode");
    }
    return tok_->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer not loaded, cannot decode");
    }
    if (token_ids.empty()) return "";

    constexpr int threshold = tt::config::SPECIAL_TOKEN_DECODE_THRESHOLD;
    if (threshold > 0) {
        std::vector<int> filtered;
        filtered.reserve(token_ids.size());
        for (int id : token_ids) {
            if (id < threshold) filtered.push_back(id);
        }
        if (filtered.empty()) return "";
        return tok_->Decode(filtered);
    }
    return tok_->Decode(token_ids);
}

// ---------------------------------------------------------------------------
// Chat template: compile-time selection between Llama 3.1 and DeepSeek V3
// ---------------------------------------------------------------------------

#ifdef MODEL_DEEPSEEK_V3

namespace {
// DeepSeek V3 uses full-width vertical line U+FF5C (｜) as delimiter
const char* USER_TAG = "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>";
const char* ASSISTANT_TAG = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";
}  // namespace

std::string Tokenizer::apply_chat_template(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool add_generation_prompt) {
    static TokenizerConfig cfg = get_tokenizer_config();

    std::ostringstream out;

    if (cfg.add_bos_token) out << cfg.bos_token;

    for (const auto& m : messages) {
        if (m.role == "system") out << m.content;
    }

    for (const auto& m : messages) {
        if (m.role == "system") continue;
        if (m.role == "user") {
            out << USER_TAG << m.content;
        } else if (m.role == "assistant") {
            out << ASSISTANT_TAG << m.content;
            if (cfg.add_eos_token) out << cfg.eos_token;
        }
    }

    if (add_generation_prompt) {
        out << ASSISTANT_TAG;
    }
    return out.str();
}

#else  // Default: Llama 3.1 8B

namespace {
const char* HEADER_START = "<|start_header_id|>";
const char* HEADER_END = "<|end_header_id|>";
const char* EOT = "<|eot_id|>";
const char* SYSTEM_PREAMBLE =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";
}  // namespace

std::string Tokenizer::apply_chat_template(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool add_generation_prompt) {
    static TokenizerConfig cfg = get_tokenizer_config();

    std::ostringstream out;

    std::string system_content;
    for (const auto& m : messages) {
        if (m.role == "system") {
            if (!system_content.empty()) system_content += "\n\n";
            system_content += m.content;
        }
    }

    if (cfg.add_bos_token) out << cfg.bos_token;

    out << HEADER_START << "system" << HEADER_END << "\n\n"
        << SYSTEM_PREAMBLE
        << system_content
        << EOT;

    for (const auto& m : messages) {
        if (m.role == "system") continue;
        std::string role = m.role.empty() ? "user" : m.role;
        out << HEADER_START << role << HEADER_END << "\n\n"
            << m.content << EOT;
    }

    if (add_generation_prompt) {
        out << HEADER_START << "assistant" << HEADER_END << "\n\n";
    }
    return out.str();
}

#endif

}  // namespace tt::utils
