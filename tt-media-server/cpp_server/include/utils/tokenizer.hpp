// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <tokenizers_cpp.h>

#include "domain/chat_message.hpp"

namespace tt::utils {

/**
 * Parsed tokenizer_config.json (Hugging Face format).
 * Token fields may be plain strings or AddedToken {"content": "..."}; load_tokenizer_config normalizes to strings.
 */
struct TokenizerConfig {
    std::string bos_token;
    std::string eos_token;
    std::string pad_token;
    std::string unk_token;
    std::string chat_template;  // Raw Jinja2 string; rendering is format-specific elsewhere
    bool add_bos_token = true;   // If true, prepend bos_token when applying chat template
    bool add_eos_token = false;  // If true, append eos_token after assistant turns
};

/**
 * Load and parse tokenizer_config.json from path (e.g. tokenizers/tokenizer_config.json).
 * Extracts bos_token, eos_token, pad_token, unk_token (from AddedToken or string), chat_template,
 * add_bos_token and add_eos_token.
 * @return true if file was read and parsed, false if file missing or invalid (out left unchanged on false).
 */
bool load_tokenizer_config(const std::string& path, TokenizerConfig& out);

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

    /**
     * Apply chat template using tokenizer_config.json (HF-style).
     * Requires tokenizer_config.json to be loadable; uses bos_token, eos_token, add_bos_token
     * and add_eos_token from config. Renders a built-in ChatML-style format (system, then
     * user/assistant turns with <<|User|>> / <<|Assistant|>>). Throws if config is missing,
     * invalid, or if add_bos_token/add_eos_token are true but the corresponding token is empty.
     */
    static std::string apply_chat_template(const std::vector<tt::domain::ChatMessage>& messages,
        bool add_generation_prompt = true);

    /** Check if tokenizer is loaded and ready. */
    bool is_loaded() const;

private:
    explicit Tokenizer(const std::string& path);
    ~Tokenizer() = default;

    std::unique_ptr<tokenizers::Tokenizer> tok_;
};

}  // namespace tt::utils
