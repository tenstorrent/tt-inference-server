// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tokenizers {

/**
 * Generic HuggingFace CPU tokenizer.
 *
 * Reads chat template from tokenizer_config.json (jinja format).
 * Stop token IDs are read from the tokenizer config or env var.
 * This avoids hardcoding model-specific templates in C++.
 */
class HFCPUTokenizer final : public Tokenizer {
 public:
  using Tokenizer::Tokenizer;

  std::string modelName() const;
  std::vector<int64_t> stopTokenIds() const;

  std::string applyChatTemplate(
      const std::vector<tt::domain::llm::ChatMessage>& messages,
      bool addGenerationPrompt,
      const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools =
          std::nullopt,
      bool enableReasoning = true, bool skipApplyChatTemplate = false) const;
};

}  // namespace tt::utils::tokenizers
