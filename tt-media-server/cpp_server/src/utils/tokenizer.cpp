// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <tokenizers_cpp.h>

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

    std::unique_ptr<tokenizers::Tokenizer> tok;
    if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".json") == 0) {
        tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    } else if (path.size() >= 7 && path.compare(path.size() - 7, 7, ".model") == 0) {
        tok = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
    } else {
        std::cerr << "[TokenizerUtil] Unknown extension; use .json or .model: " << path << std::endl;
        return;
    }

    if (!tok) {
        std::cerr << "[TokenizerUtil] Failed to create tokenizer from: " << path << std::endl;
        return;
    }

    impl_ = std::make_unique<TokenizerUtilImpl>();
    impl_->tok = std::move(tok);
}

TokenizerUtil::~TokenizerUtil() = default;

TokenizerUtil::TokenizerUtil(TokenizerUtil&&) noexcept = default;

TokenizerUtil& TokenizerUtil::operator=(TokenizerUtil&&) noexcept = default;

bool TokenizerUtil::is_loaded() const {
    return impl_ && impl_->tok;
}

std::vector<int> TokenizerUtil::encode(const std::string& text) const {
    if (impl_ && impl_->tok) {
        return impl_->tok->Encode(text);
    }
    return {};
}

std::string TokenizerUtil::decode(const std::vector<int>& token_ids) const {
    if (impl_ && impl_->tok) {
        return impl_->tok->Decode(token_ids);
    }
    return "";
}

}  // namespace tt::utils
