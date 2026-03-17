// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "utils/tokenizer.hpp"

namespace tt::utils {

class DeepseekTokenizer final : public Tokenizer {
 public:
  using Tokenizer::Tokenizer;

  std::string modelName() const override {
    return "deepseek-ai/DeepSeek-R1-0528";
  }
  int specialTokenDecodeThreshold() const override { return -1; }
  std::vector<int64_t> stopTokenIds() const override { return {1}; }

  std::string applyChatTemplate(
      const std::vector<domain::ChatMessage>& messages,
      bool addGenerationPrompt) const override;
};

}  // namespace tt::utils
