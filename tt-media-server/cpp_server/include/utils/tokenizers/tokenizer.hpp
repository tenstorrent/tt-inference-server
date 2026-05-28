// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <tokenizers_cpp.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "config/types.hpp"
#include "domain/llm/chat_message.hpp"
#include "domain/tool_calls/tool.hpp"

namespace tt::utils::tokenizers {

using namespace tt::domain::llm;

/**
 * Parsed tokenizer_config.json (Hugging Face format).
 * Token fields may be plain strings or AddedToken {"content": "..."}; parsing
 * normalizes to strings.
 */
struct TokenizerConfig {
  std::string bos_token;
  std::string eos_token;
  std::string pad_token;
  std::string unk_token;
  std::string chat_template;  // Raw Jinja2 string; rendering is format-specific
                              // elsewhere
  bool add_bos_token =
      true;  // If true, prepend bos_token when applying chat template
  bool add_eos_token =
      false;  // If true, append eos_token after assistant turns
};

/**
 * Load tokenizer config from the path given by config::tokenizer_config_path(),
 * validate add_bos_token/add_eos_token vs bos_token/eos_token, and return the
 * config. The no-arg overload caches the result (global singleton, first call
 * wins). The path overload always loads fresh from the given file.
 * @throws std::runtime_error if config path is empty, file cannot be loaded, or
 * tokens are missing when flags are set.
 */
TokenizerConfig getTokenizerConfig();
TokenizerConfig getTokenizerConfig(const std::string& configPath);

/**
 * Tokenizer utility wrapping mlc-ai/tokenizers-cpp (HuggingFace /
 * SentencePiece). The underlying Rust tokenizer is not thread-safe and a single
 * instance must not be shared across threads. Use `activeTokenizer()` to obtain
 * a thread-local instance.
 *
 * Model-specific behavior (chat template format, special token decode
 * filtering, stop tokens) is provided by subclasses: DeepseekTokenizer and
 * LlamaTokenizer.
 */
class Tokenizer {
 public:
  /**
   * Construct a tokenizer from a .json (HuggingFace) or .model (SentencePiece)
   * file.
   * @throws std::runtime_error if path is empty, file is unreadable, or format
   * is unsupported.
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
   * Decode token IDs to text.
   * @param skipSpecialTokens If true (default), special tokens (parsed from
   *   the tokenizer JSON's added_tokens with "special": true) are filtered out
   *   before decoding. If false, all tokens are decoded as-is.
   * @throws std::runtime_error if tokenizer not loaded.
   */
  std::string decode(const std::vector<int>& tokenIds,
                     bool skipSpecialTokens = true) const;

  /** Check if tokenizer is loaded and ready. */
  bool isLoaded() const;

  virtual std::string modelName() const = 0;
  virtual std::vector<int64_t> stopTokenIds() const = 0;

  /**
   * Token id sequence that ends an assistant generation prompt in the
   * model's chat template (e.g. Llama-3
   * `<|start_header_id|>assistant<|end_header_id|>\n\n`, DeepSeek
   * `<｜Assistant｜>`).
   *
   * Used by `computePrefixCachingInfoFromTokens` to locate turn boundaries
   * in a pre-tokenized prompt without round-tripping through text. The
   * count of occurrences in the prompt equals the number of assistant
   * turns (including the trailing one we are about to generate); the
   * second-to-last occurrence marks the cache-lookup boundary.
   *
   * Returns an empty vector when the tokenizer does not expose a stable
   * assistant marker; callers must then treat every request as fresh.
   */
  virtual std::vector<int> assistantHeaderSequence() const { return {}; }

  /**
   * Apply the model-specific chat template to a list of messages.
   * @param enableReasoning When false, reasoning models (e.g. DeepSeek-R1)
   *   inject a closed think block to suppress chain-of-thought output.
   * @param skipApplyChatTemplate When true, skip adding <bos><user> and
   *   <assistant> tags (returns raw message content only).
   */
  virtual std::string applyChatTemplate(
      const std::vector<tt::domain::llm::ChatMessage>& messages,
      bool addGenerationPrompt = true,
      const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools =
          std::nullopt,
      bool enableReasoning = true,
      bool skipApplyChatTemplate = false) const = 0;

  /**
   * Stream decoder for incremental token-by-token decoding.
   * Buffers incomplete UTF-8 sequences across tokens automatically.
   */
  class StreamDecoder {
   public:
    explicit StreamDecoder(const Tokenizer& tokenizer,
                           bool skipSpecialTokens = true);

    /**
     * Decodes the next token. Returns the decoded text delta, or "" if the
     * token is part of an incomplete multi-byte UTF-8 sequence still being
     * buffered.
     */
    std::string step(int tokenId);

    /**
     * Flush any remaining buffered tokens (call on final token).
     * Returns whatever text the buffer decodes to, even if it contains
     * replacement characters.
     */
    std::string flush();

   private:
    const Tokenizer& tokenizer_;
    std::vector<int> pending_;
    bool skipSpecialTokens_;
  };

  std::unique_ptr<StreamDecoder> createStreamDecoder(
      bool skipSpecialTokens = true) const;

  std::vector<std::string> getEncodedVocab() const;

 protected:
  std::unique_ptr<::tokenizers::Tokenizer> tok_;
  TokenizerConfig cfg_;
  std::unordered_set<int> specialTokenIds_;
};

/**
 * Factory: create a Tokenizer for the given model, loading from path.
 * DEEPSEEK_R1_0528 -> DeepseekTokenizer
 * LLAMA_3_1_8B_INSTRUCT -> LlamaTokenizer
 * KIMI_K2_6 -> DeepseekTokenizer (temporary behavior)
 */
std::unique_ptr<Tokenizer> createTokenizer(config::ModelType model,
                                           const std::string& path);

/**
 * Tokenizer directory name for a given model type. Used to resolve tokenizer
 * file paths before a Tokenizer instance exists.
 */
std::string tokenizerDirForModel(config::ModelType model);

/**
 * Active tokenizer for the calling thread, auto-initialized from
 * LLM_DEVICE_BACKEND on first access (per thread). Each thread gets its own
 * instance so encode/decode are race-free without locking. The reference is
 * only valid on the calling thread; do not capture it for cross-thread use.
 *
 * Instantiation parses tokenizer.json synchronously and is expensive on
 * large vocabs. For model-level constants used on the request hot path
 * prefer `staticInfoFor()` below.
 */
const Tokenizer& activeTokenizer();

/**
 * Per-model constants that don't require a live Tokenizer instance.
 * Lets the request hot path read modelName / stopTokenIds /
 * assistantHeaderSequence without parsing tokenizer.json.
 */
struct StaticTokenizerInfo {
  std::string_view modelName;
  std::vector<int64_t> stopTokenIds;
  std::vector<int> assistantHeaderSequence;
};

/**
 * Static constants for `model`. Throws std::invalid_argument if no entry
 * is registered. O(1) and thread-safe; never touches the tokenizer.
 */
const StaticTokenizerInfo& staticInfoFor(config::ModelType model);

/// Shorthand for `staticInfoFor(config::modelType())`.
const StaticTokenizerInfo& staticInfo();

}  // namespace tt::utils::tokenizers
