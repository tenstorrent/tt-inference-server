// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tokenizers {

// Derive whether generation should begin inside a reasoning block (unclosed
// <think> before the first decoded token). Scans backward for the
// last think marker; if none is found, returns `initial_in_thinking`. Mirrors
// TokenCommitter / conversation_hasher marker rules.
inline bool computeThinkingPhaseFromTokens(bool initial_in_thinking,
                                           std::span<const int> tokens,
                                           int64_t think_open,
                                           int64_t think_end) {
  if (think_open == kNoTokenId && think_end == kNoTokenId) {
    return false;
  }
  for (size_t i = tokens.size(); i-- > 0;) {
    const int tok = tokens[i];
    if (think_open != kNoTokenId && tok == static_cast<int>(think_open)) {
      return true;
    }
    if (think_end != kNoTokenId && tok == static_cast<int>(think_end)) {
      return false;
    }
  }
  return initial_in_thinking;
}

// Scan the prompt tokens that will be handed to DecodeScheduler, clamped to
// maxISL() so the result matches PromptTable::store truncation.
inline bool computeStartsInThinkingForDecodePrompt(
    std::span<const int> tokens, bool initial_in_thinking = false) {
  const auto [think_open, think_end] = thinkTokenIds();
  const size_t clamped = std::min(tokens.size(), tt::config::maxISL());
  return computeThinkingPhaseFromTokens(
      initial_in_thinking, tokens.subspan(0, clamped), think_open, think_end);
}

// Recompute LLMRequest::starts_in_thinking from the current prompt token
// buffer (full prompt or post-delta-trim remainder).
inline void refreshStartsInThinking(tt::domain::llm::LLMRequest& req,
                                    bool initial_in_thinking = false) {
  if (auto* tokens = std::get_if<std::vector<int>>(&req.prompt)) {
    req.starts_in_thinking = computeStartsInThinkingForDecodePrompt(
        std::span<const int>(*tokens), initial_in_thinking);
    return;
  }
  req.starts_in_thinking = false;
}

}  // namespace tt::utils::tokenizers
