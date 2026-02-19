// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <tokenizers_cpp.h>

#include "domain/chat_message.hpp"

namespace tt::utils {

using namespace std;

/**
 * Parsed tokenizer_config.json (Hugging Face format).
 * Token fields may be plain strings or AddedToken {"content": "..."}; parsing normalizes to strings.
 */
struct TokenizerConfig {
    string bos_token;
    string eos_token;
    string pad_token;
    string unk_token;
    string chat_template;  // Raw Jinja2 string; rendering is format-specific elsewhere
    bool add_bos_token = true;   // If true, prepend bos_token when applying chat template
    bool add_eos_token = false;  // If true, append eos_token after assistant turns
};

/**
 * Load tokenizer config from the path given by config::tokenizer_config_path(), validate
 * add_bos_token/add_eos_token vs bos_token/eos_token, and return the config.
 * @throws runtime_error if config path is empty, file cannot be loaded, or tokens are missing when flags are set.
 */
TokenizerConfig get_tokenizer_config();

/**
 * Tokenizer utility wrapping mlc-ai/tokenizers-cpp (HuggingFace / SentencePiece).
 * Each instance owns its own underlying tokenizer, so separate instances are safe
 * to use from different threads without synchronization.
 */
class Tokenizer {
public:
    /**
     * Construct a tokenizer from a .json (HuggingFace) or .model (SentencePiece) file.
     * @throws runtime_error if path is empty, file is unreadable, or format is unsupported.
     */
    explicit Tokenizer(const string& path);
    ~Tokenizer() = default;

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;

    /**
     * Encode text to token IDs.
     * @throws runtime_error if tokenizer not loaded.
     */
    vector<int> encode(const string& text) const;

    /**
     * Decode token IDs to text.
     * @throws runtime_error if tokenizer not loaded.
     */
    string decode(const vector<int>& token_ids) const;

    /**
     * Apply chat template using tokenizer_config.json (HF-style).
     * Requires tokenizer_config.json to be loadable; uses bos_token, eos_token, add_bos_token
     * and add_eos_token from config. Renders a built-in ChatML-style format (system, then
     * user/assistant turns with <<|User|>> / <<|Assistant|>>). Throws if config is missing,
     * invalid, or if add_bos_token/add_eos_token are true but the corresponding token is empty.
     */
    static string apply_chat_template(const vector<tt::domain::ChatMessage>& messages,
        bool add_generation_prompt = true);

    /** Check if tokenizer is loaded and ready. */
    bool is_loaded() const;

private:
    unique_ptr<tokenizers::Tokenizer> tok_;
};

}  // namespace tt::utils
