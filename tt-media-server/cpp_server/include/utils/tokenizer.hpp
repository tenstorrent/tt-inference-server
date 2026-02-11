// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>

#ifdef ENABLE_TOKENIZER
#include "utils/tokenizer_impl.hpp"
#endif

namespace tt::utils {

#ifndef ENABLE_TOKENIZER
struct TokenizerUtilImpl {};
#endif

/**
 * Tokenizer utility wrapping mlc-ai/tokenizers-cpp (HuggingFace / SentencePiece).
 * Used for encode (text -> token IDs) in pre_process and decode (token IDs -> text)
 * when returning results (vLLM-style).
 */
class TokenizerUtil {
public:
    TokenizerUtil() = default;
    ~TokenizerUtil();
    TokenizerUtil(TokenizerUtil&&) noexcept;
    TokenizerUtil& operator=(TokenizerUtil&&) noexcept;
    TokenizerUtil(const TokenizerUtil&) = delete;
    TokenizerUtil& operator=(const TokenizerUtil&) = delete;

    /**
     * Load tokenizer from file path.
     * Supports tokenizer.json (HuggingFace) or tokenizer.model (SentencePiece).
     * @param path Path to tokenizer file; empty = no-op.
     * @return New instance with tokenizer loaded, or empty instance on failure.
     */
    static TokenizerUtil load(const std::string& path);

    /** True if a tokenizer is loaded and ready. */
    bool is_loaded() const;

    /** Encode text to token IDs. Returns empty vector if not loaded. */
    std::vector<int> encode(const std::string& text) const;

    /** Decode token IDs to text. Returns empty string if not loaded. */
    std::string decode(const std::vector<int>& token_ids) const;

private:
    std::unique_ptr<TokenizerUtilImpl> impl_;
};

}  // namespace tt::utils
