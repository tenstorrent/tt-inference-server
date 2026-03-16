// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <tokenizers_cpp.h>

#include "config/constants.hpp"
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
 * The no-arg overload caches the result (global singleton, first call wins).
 * The path overload always loads fresh from the given file.
 * @throws std::runtime_error if config path is empty, file cannot be loaded, or tokens are missing when flags are set.
 */
TokenizerConfig get_tokenizer_config();
TokenizerConfig get_tokenizer_config(const std::string& config_path);

/**
 * Tokenizer utility wrapping mlc-ai/tokenizers-cpp (HuggingFace / SentencePiece).
 * Each instance owns its own underlying tokenizer, so separate instances are safe
 * to use from different threads without synchronization.
 *
 * Model-specific behavior (chat template format, special token decode filtering, stop tokens)
 * is provided by subclasses: DeepseekTokenizer and LlamaTokenizer.
 */
class Tokenizer {
public:
    /**
     * Construct a tokenizer from a .json (HuggingFace) or .model (SentencePiece) file.
     * @throws std::runtime_error if path is empty, file is unreadable, or format is unsupported.
     */
    explicit Tokenizer(const std::string& path);
    virtual ~Tokenizer() = default;

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
     * Decode token IDs to text. When the model defines a special-token decode
     * threshold (e.g. Llama 3 >= 128000), tokens at or above that ID are filtered out.
     * @throws std::runtime_error if tokenizer not loaded.
     */
    std::string decode(const std::vector<int>& token_ids) const;

    /** Check if tokenizer is loaded and ready. */
    bool is_loaded() const;

    virtual std::string model_name() const = 0;
    virtual int special_token_decode_threshold() const = 0;
    virtual std::vector<int64_t> stop_token_ids() const = 0;

    /**
     * Apply the model-specific chat template to a list of messages.
     */
    virtual std::string apply_chat_template(
        const std::vector<tt::domain::ChatMessage>& messages,
        bool add_generation_prompt = true) const = 0;

protected:
    std::unique_ptr<tokenizers::Tokenizer> tok_;
    TokenizerConfig cfg_;
    mutable int cached_special_token_threshold_ = -2;  // -2 = unset, then special_token_decode_threshold()
};

/**
 * Factory: create a Tokenizer for the given model, loading from path.
 * DEEPSEEK_R1_0528 -> DeepseekTokenizer
 * LLAMA_3_1_8B_INSTRUCT -> LlamaTokenizer
 */
std::unique_ptr<Tokenizer> create_tokenizer(config::ModelType model, const std::string& path);

/**
 * Tokenizer directory name for a given model type. Used to resolve tokenizer
 * file paths before a Tokenizer instance exists.
 */
std::string tokenizer_dir_for_model(config::ModelType model);

/**
 * Global active tokenizer, auto-initialized from LLM_DEVICE_BACKEND on first access.
 * Thread-safe (C++11 function-local static initialization).
 * Intended for metadata access (model_name, stop_token_ids, apply_chat_template);
 * for encode/decode in multithreaded contexts, create separate instances.
 */
const Tokenizer& active_tokenizer();

}  // namespace tt::utils
