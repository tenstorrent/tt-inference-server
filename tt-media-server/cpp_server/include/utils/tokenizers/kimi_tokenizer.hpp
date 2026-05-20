// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tokenizers {

/**
 * Tokenizer for moonshotai/Kimi-K2.6.
 *
 * Kimi K2.6 ships tiktoken.model + a custom Python tokenizer class, not an HF
 * tokenizer.json that mlc-ai/tokenizers-cpp can load. This subclass calls
 * directly into Python (via pybind11::embed) using:
 *   - `tiktoken` for encode/decode — same engine the reference Python
 *     tokenization_kimi.py uses, so byte-exact parity by construction.
 *   - `jinja2` to render chat_template.jinja — full Jinja2, no minja subset
 *     gaps to worry about.
 *
 * Trade-off: every encode/decode/render acquires the GIL. tiktoken-py releases
 * the GIL inside its Rust core, so concurrent throughput is acceptable, but
 * the per-call boundary cost is ~10–30 μs higher than a native tokenizer.
 *
 * Requires Python with the `tiktoken` and `jinja2` packages available at
 * runtime (see build.sh and the project's Python deps).
 *
 * Construct with the directory holding tiktoken.model + tokenizer_config.json
 * + chat_template.jinja. createTokenizer() in tokenizer.cpp passes the
 * tiktoken.model path; we strip the filename to find the rest.
 */
class KimiTokenizer final : public Tokenizer {
 public:
  /**
   * @param tiktokenModelPath Absolute path to tokenizers/<repo>/tiktoken.model
   *   (sibling tokenizer_config.json and chat_template.jinja must exist).
   * @throws std::runtime_error if Python init fails, required files are
   *   missing, or the tiktoken/jinja2 imports fail.
   */
  explicit KimiTokenizer(const std::string& tiktokenModelPath);
  ~KimiTokenizer() override;

  KimiTokenizer(const KimiTokenizer&) = delete;
  KimiTokenizer& operator=(const KimiTokenizer&) = delete;

  std::vector<int> encode(const std::string& text) const override;
  std::string decode(const std::vector<int>& tokenIds,
                     bool skipSpecialTokens = true) const override;
  bool isLoaded() const override;

  std::string modelName() const override { return "moonshotai/Kimi-K2.6"; }
  std::vector<int64_t> stopTokenIds() const override { return {163586}; }

  std::string applyChatTemplate(
      const std::vector<tt::domain::llm::ChatMessage>& messages,
      bool addGenerationPrompt,
      const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools =
          std::nullopt,
      bool enableReasoning = true,
      bool skipApplyChatTemplate = false) const override;

 private:
  // Holds py::object members. Hidden in .cpp to keep pybind11 out of the
  // public header.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::utils::tokenizers
