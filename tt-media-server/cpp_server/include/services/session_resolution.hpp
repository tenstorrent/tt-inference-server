// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "services/session_manager.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/thinking_phase.hpp"

namespace tt::services::session_resolution {

struct DeltaPromptOptions {
  // LLMPipeline keeps the full prompt in DECODE_ONLY mode until dispatch can
  // decide whether prefill runs locally or on the prefill server.
  bool skipUnlessRegularMode = false;
  bool setKvPositionId = false;
  // Prior thinking phase from the matched session (STOP->resume inside an
  // unclosed <think> block). Ignored for fresh SUBMIT.
  bool initialInThinking = false;
  std::string_view logPrefix;
};

// Trim the cached prefix from a token prompt and update prompt_tokens_count.
// Returns the number of tokens removed.
inline uint32_t applyDeltaPrompt(tt::domain::llm::LLMRequest& req,
                                 uint32_t matchedTokens,
                                 DeltaPromptOptions options = {}) {
  if (options.skipUnlessRegularMode &&
      tt::config::llmMode() != tt::config::LLMMode::REGULAR) {
    return 0;
  }

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  const size_t skip = static_cast<size_t>(matchedTokens);
  if (skip >= tokens.size()) {
    return 0;
  }

  if (!options.logPrefix.empty() && matchedTokens > 0) {
    TT_LOG_DEBUG("{} applyDeltaPrompt: matchedTokens={} remainder={}",
                 options.logPrefix, matchedTokens,
                 static_cast<uint32_t>(tokens.size()) - matchedTokens);
  }

  tokens.erase(tokens.begin(), tokens.begin() + static_cast<ptrdiff_t>(skip));
  req.prompt_tokens_count = static_cast<int>(tokens.size());

  if (options.setKvPositionId && matchedTokens > 0) {
    req.kv_position_id = matchedTokens - 1;
  }

  tt::utils::tokenizers::refreshStartsInThinking(req,
                                                 options.initialInThinking);

  return matchedTokens;
}

struct SlotCopyPlan {
  uint32_t slotToCopyFrom;
  uint32_t matchedTokens = 0;
};

// Pick and lock a source slot for KV-copy allocation, if a candidate is worth
// copying. The caller owns unlocking slotToCopyFrom after createSession
// returns.
std::optional<SlotCopyPlan> prepareSlotCopy(
    SessionManager& sessionManager,
    const std::vector<SessionManager::Candidate>& candidates, uint32_t taskId,
    std::string_view logPrefix);

}  // namespace tt::services::session_resolution
