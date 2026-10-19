// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <utility>

#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tokenizers {

class LlamaTokenizer final : public Tokenizer {
 public:
  explicit LlamaTokenizer(const std::string& path,
                          std::string model = "meta-llama/Llama-3.1-8B-Instruct")
      : Tokenizer(path), modelId(std::move(model)) {}

  std::string modelName() const override { return modelId; }
  std::vector<uint32_t> stopTokenIds() const {
    return {128001, 128008, 128009};
  }

  // The assistant generation prompt is the multi-token sequence
  // `<|start_header_id|>assistant<|end_header_id|>\n\n`. Encoding it once
  // and matching the resulting id sequence in inbound prompts is what
  // anchors token-level prefix caching for Llama-3 chat templates.
  std::vector<uint32_t> assistantHeaderSequence() const override;

  std::string applyChatTemplate(
      const std::vector<tt::domain::llm::ChatMessage>& messages,
      bool addGenerationPrompt, bool enableReasoning = true,
      bool skipApplyChatTemplate = false) const;

 private:
  std::string modelId;
};

}  // namespace tt::utils::tokenizers
