// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <tokenizers_cpp.h>

namespace tt::utils {

/**
 * Tokenizer utility wrapping mlc-ai/tokenizers-cpp (HuggingFace / SentencePiece).
 * Used for encode (text -> token IDs) in pre_process and decode (token IDs -> text)
 * when returning results (vLLM-style).
 */
class TokenizerUtil {
public:
    /**
     * Load tokenizer from file path.
     * Supports tokenizer.json (HuggingFace) or tokenizer.model (SentencePiece).
     * @param path Path to tokenizer file; empty = no-op (no tokenizer loaded).
     */
    explicit TokenizerUtil(const std::string& path = "");
    ~TokenizerUtil();
    TokenizerUtil(TokenizerUtil&&) noexcept;
    TokenizerUtil& operator=(TokenizerUtil&&) noexcept;
    TokenizerUtil(const TokenizerUtil&) = delete;
    TokenizerUtil& operator=(const TokenizerUtil&) = delete;

    /** Encode text to token IDs. Returns empty vector if tokenizer not loaded. */
    std::vector<int> encode(const std::string& text) const;

    /** Decode token IDs to text. Returns empty string if tokenizer not loaded. */
    std::string decode(const std::vector<int>& token_ids) const;

private:
    std::unique_ptr<tokenizers::Tokenizer> tok_;
};

}  // namespace tt::utils
