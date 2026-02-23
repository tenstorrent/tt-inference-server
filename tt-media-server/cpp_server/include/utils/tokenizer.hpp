// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <tokenizers_cpp.h>

#include "config/model_config.hpp"
#include "domain/chat_message.hpp"

namespace tt::utils {

/**
 * Parsed tokenizer_config.json (Hugging Face format).
 * Token fields may be plain strings or AddedToken {"content": "..."}; parsing normalizes to strings.
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
 * Load tokenizer config from the path given by config::tokenizer_config_path(), validate
 * add_bos_token/add_eos_token vs bos_token/eos_token, and return the config.
 * @throws std::runtime_error if config path is empty, file cannot be loaded, or tokens are missing when flags are set.
 */
TokenizerConfig get_tokenizer_config();

/**
 * Tokenizer utility wrapping mlc-ai/tokenizers-cpp (HuggingFace / SentencePiece).
 * Each instance owns its own underlying tokenizer, so separate instances are safe
 * to use from different threads without synchronization.
 *
 * Model-specific behavior (chat template format, special token decode filtering)
 * is selected at compile time via MODEL_TYPE — see config/model_config.hpp.
 */
class Tokenizer {
public:
    /**
     * Construct a tokenizer from a .json (HuggingFace) or .model (SentencePiece) file.
     * @throws std::runtime_error if path is empty, file is unreadable, or format is unsupported.
     */
    explicit Tokenizer(const std::string& path);
    ~Tokenizer() = default;

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;

    /**
     * Encode text to token IDs.
     * @throws std::runtime_error if tokenizer not loaded.
     */
    std::vector<int> encode(const std::string& text) const;

    /**
     * Decode token IDs to text. When the active model defines a special-token decode
     * threshold (e.g. Llama 3 >= 128000), tokens at or above that ID are filtered out.
     * @throws std::runtime_error if tokenizer not loaded.
     */
    std::string decode(const std::vector<int>& token_ids) const;

    /**
     * Apply the chat template for the compile-time selected model.
     * Llama 3.1 8B: header/eot tags with a knowledge-cutoff preamble.
     * DeepSeek V3: Unicode-delimited User/Assistant tags without a preamble.
     */
    static std::string apply_chat_template(const std::vector<tt::domain::ChatMessage>& messages,
        bool add_generation_prompt = true);

    /** Check if tokenizer is loaded and ready. */
    bool is_loaded() const;

private:
    std::unique_ptr<tokenizers::Tokenizer> tok_;
};

}  // namespace tt::utils
