// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tokenizers {

class LlamaTokenizer final : public Tokenizer {
 public:
  using Tokenizer::Tokenizer;

  std::string modelName() const { return "meta-llama/Llama-3.1-8B-Instruct"; }
  std::vector<int64_t> stopTokenIds() const { return {128001, 128008, 128009}; }

  // The assistant generation prompt is the multi-token sequence
  // `<|start_header_id|>assistant<|end_header_id|>\n\n`. Encoding it once
  // and matching the resulting id sequence in inbound prompts is what
  // anchors token-level prefix caching for Llama-3 chat templates.
  std::vector<int> assistantHeaderSequence() const override;

  std::string applyChatTemplate(
      const std::vector<tt::domain::llm::ChatMessage>& messages,
      bool addGenerationPrompt,
      const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools =
          std::nullopt,
      bool enableReasoning = true, bool skipApplyChatTemplate = false) const;
};

}  // namespace tt::utils::tokenizers
