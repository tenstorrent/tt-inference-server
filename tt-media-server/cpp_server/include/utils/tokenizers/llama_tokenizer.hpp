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

  // <|eot_id|> = 128009 marks the end of every header-id block, including
  // assistant turns; that is the prior-turn boundary used by token-level
  // prefix caching.
  int turnBoundaryTokenId() const override { return 128009; }

  std::string applyChatTemplate(
      const std::vector<tt::domain::llm::ChatMessage>& messages,
      bool addGenerationPrompt,
      const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools =
          std::nullopt,
      bool enableReasoning = true, bool skipApplyChatTemplate = false) const;
};

}  // namespace tt::utils::tokenizers
