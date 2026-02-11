// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

#ifdef ENABLE_TOKENIZER
#include <tokenizers_cpp.h>
#endif

namespace tt::utils {

// Destructor and move ops; impl_ complete type is in tokenizer_impl.hpp when ENABLE_TOKENIZER.
TokenizerUtil::~TokenizerUtil() = default;

TokenizerUtil::TokenizerUtil(TokenizerUtil&&) noexcept = default;

TokenizerUtil& TokenizerUtil::operator=(TokenizerUtil&&) noexcept = default;

TokenizerUtil TokenizerUtil::load(const std::string& path) {
    TokenizerUtil out;
    if (path.empty()) {
        return out;
    }

#ifdef ENABLE_TOKENIZER
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[TokenizerUtil] Failed to open: " << path << std::endl;
        return out;
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
        return out;
    }

    if (!tok) {
        std::cerr << "[TokenizerUtil] Failed to create tokenizer from: " << path << std::endl;
        return out;
    }

    out.impl_ = std::make_unique<TokenizerUtilImpl>();
    out.impl_->tok = std::move(tok);
#endif

    return out;
}

bool TokenizerUtil::is_loaded() const {
#ifdef ENABLE_TOKENIZER
    return impl_ && impl_->tok;
#else
    return impl_ != nullptr;
#endif
}

std::vector<int> TokenizerUtil::encode(const std::string& text) const {
#ifdef ENABLE_TOKENIZER
    if (impl_ && impl_->tok) {
        return impl_->tok->Encode(text);
    }
#else
    (void)text;
#endif
    return {};
}

std::string TokenizerUtil::decode(const std::vector<int>& token_ids) const {
#ifdef ENABLE_TOKENIZER
    if (impl_ && impl_->tok) {
        return impl_->tok->Decode(token_ids);
    }
#else
    (void)token_ids;
#endif
    return "";
}

}  // namespace tt::utils
