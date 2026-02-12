// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "config/settings.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace tt::utils {

Tokenizer& Tokenizer::instance(const std::string& path) {
    static Tokenizer instance(path);
    return instance;
}

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
    return tok_->Decode(token_ids);
}

namespace {

const char* DEFAULT_BOS = "\n";
const char* USER_TAG = "<<|User|>>";
const char* ASSISTANT_TAG = "<<|Assistant|>>";
const char* DEFAULT_EOS = "<<|end▁of▁sentence|>>";

}  // namespace

std::string Tokenizer::apply_chat_template(const std::vector<tt::domain::ChatMessage>& messages,
    bool add_generation_prompt) {
    TokenizerConfig cfg;
    std::string config_path = tt::config::tokenizer_config_path();
    if (config_path.empty()) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer config not found (tokenizer_config.json missing)");
    }
    load_tokenizer_config(config_path, cfg);
    std::string bos = cfg.bos_token.empty() ? DEFAULT_BOS : cfg.bos_token;
    std::string eos = cfg.eos_token.empty() ? std::string(DEFAULT_EOS) : cfg.eos_token;

    std::ostringstream out;
    std::string system_prompt;
    bool first_system = true;
    for (const auto& m : messages) {
        if (m.role != "system") continue;
        if (!first_system) system_prompt += "\n\n";
        system_prompt += m.content;
        first_system = false;
    }
    out << bos << system_prompt;
    for (const auto& m : messages) {
        std::string role = m.role.empty() ? "user" : m.role;
        if (role == "system") continue;
        if (role == "user") {
            out << USER_TAG << m.content;
        } else if (role == "assistant") {
            out << ASSISTANT_TAG << m.content << eos;
        }
    }
    if (add_generation_prompt) {
        out << ASSISTANT_TAG;
    }
    return out.str();
}

}  // namespace tt::utils
