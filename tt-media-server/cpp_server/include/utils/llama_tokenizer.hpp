// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "utils/tokenizer.hpp"

namespace tt::utils {

class LlamaTokenizer final : public Tokenizer {
 public:
  using Tokenizer::Tokenizer;

  std::string modelName() const { return "meta-llama/Llama-3.1-8B-Instruct"; }
  std::vector<int64_t> stopTokenIds() const { return {128001, 128008, 128009}; }

  std::string applyChatTemplate(
      const std::vector<domain::ChatMessage>& messages,
      bool addGenerationPrompt = true,
      const std::optional<std::vector<domain::Tool>>& tools = std::nullopt) const override;
};

}  // namespace tt::utils
