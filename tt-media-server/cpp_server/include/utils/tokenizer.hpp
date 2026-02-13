// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <tokenizers_cpp.h>

namespace tt::utils {

/**
 * Tokenizer utility wrapping mlc-ai/tokenizers-cpp (HuggingFace / SentencePiece).
 * Singleton pattern - loads tokenizer once and reuses it across the application.
 * Thread-safe initialization (C++11 guarantees).
 */
class Tokenizer {
public:
    /**
     * Get the singleton instance.
     * @param path Path to tokenizer file (only used on first call).
     * @return Reference to the singleton instance.
     */
    static Tokenizer& instance(const std::string& path = "");

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = delete;
    Tokenizer& operator=(Tokenizer&&) = delete;

    /**
     * Encode text to token IDs.
     * @throws std::runtime_error if tokenizer not loaded.
     */
    std::vector<int> encode(const std::string& text) const;

    /**
     * Decode token IDs to text.
     * @throws std::runtime_error if tokenizer not loaded.
     */
    std::string decode(const std::vector<int>& token_ids) const;

    /** Check if tokenizer is loaded and ready. */
    bool is_loaded() const;

private:
    explicit Tokenizer(const std::string& path);
    ~Tokenizer() = default;

    std::unique_ptr<tokenizers::Tokenizer> tok_;
};

}  // namespace tt::utils
