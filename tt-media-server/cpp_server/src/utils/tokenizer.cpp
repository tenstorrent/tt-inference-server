// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace tt::utils {

TokenizerUtil& TokenizerUtil::instance(const std::string& path) {
    static TokenizerUtil instance(path);
    return instance;
}

TokenizerUtil::TokenizerUtil(const std::string& path) {
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

std::vector<int> TokenizerUtil::encode(const std::string& text) const {
    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer not loaded, cannot encode");
    }
    return tok_->Encode(text);
}

std::string TokenizerUtil::decode(const std::vector<int>& token_ids) const {
    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer not loaded, cannot decode");
    }
    return tok_->Decode(token_ids);
}

}  // namespace tt::utils
