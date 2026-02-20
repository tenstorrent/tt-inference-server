// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

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

    std::cout << "[TokenizerUtil] Loaded tokenizer from: " << path << std::endl;
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
    // Llama 3 special tokens occupy IDs [128000, 128255]. They must never appear
    // in decoded output — filter them before passing to the underlying tokenizer.
    std::vector<int> filtered;
    filtered.reserve(token_ids.size());
    for (int id : token_ids) {
        if (id < SPECIAL_TOKEN_START) filtered.push_back(id);
    }
    if (filtered.empty()) return "";
    return tok_->Decode(filtered);
}

namespace {

constexpr const char* HEADER_START = "<|start_header_id|>";
constexpr const char* HEADER_END = "<|end_header_id|>";
constexpr const char* EOT = "<|eot_id|>";
constexpr const char* SYSTEM_PREAMBLE =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

}  // namespace

std::string Tokenizer::apply_chat_template(const std::vector<tt::domain::ChatMessage>& messages,
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

}  // namespace tt::utils
