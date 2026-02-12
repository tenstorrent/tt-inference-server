// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace tt::utils {

TokenizerUtil::TokenizerUtil(const std::string& path) {
    if (path.empty()) {
        return;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[TokenizerUtil] Failed to open: " << path << std::endl;
        return;
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
        std::cerr << "[TokenizerUtil] Unknown extension; use .json or .model: " << path << std::endl;
        return;
    }

    if (!tok_) {
        std::cerr << "[TokenizerUtil] Failed to create tokenizer from: " << path << std::endl;
    }
}

TokenizerUtil::~TokenizerUtil() = default;

TokenizerUtil::TokenizerUtil(TokenizerUtil&&) noexcept = default;

TokenizerUtil& TokenizerUtil::operator=(TokenizerUtil&&) noexcept = default;

std::vector<int> TokenizerUtil::encode(const std::string& text) const {
    if (tok_) {
        return tok_->Encode(text);
    }
    return {};
}

std::string TokenizerUtil::decode(const std::vector<int>& token_ids) const {
    if (tok_) {
        return tok_->Decode(token_ids);
    }
    return "";
}

}  // namespace tt::utils
