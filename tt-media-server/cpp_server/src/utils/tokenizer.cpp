// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "utils/tokenizer_strategy.hpp"

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

    const auto& strategy = active_tokenizer_strategy();
    std::cout << "[TokenizerUtil] Loaded tokenizer from: " << path
              << " (model: " << strategy.model_name() << ")" << std::endl;
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

    int threshold = active_tokenizer_strategy().special_token_decode_threshold();
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

std::string Tokenizer::apply_chat_template(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool add_generation_prompt) {
    static TokenizerConfig cfg = get_tokenizer_config();
    return active_tokenizer_strategy().apply_chat_template(messages, add_generation_prompt, cfg);
}

}  // namespace tt::utils
