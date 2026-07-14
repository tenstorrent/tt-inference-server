// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <trantor/net/EventLoop.h>

#include "domain/sentinel_values.hpp"
#include "services/session_manager.hpp"

namespace tt::services::decode_slot_reservation {

/** Decode-side KV slot grant returned to prefill after reservation. */
struct DecodeDestinationSlot {
  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  std::string sessionId;
  int decodePositionId = 0;
  int decodeSkipTokens = 0;
  bool continuation = false;
  int accumulatedThinkTokens = 0;
};

struct ResolveInput {
  uint32_t taskId = 0;
  std::vector<uint64_t> registrationHashes;
  std::optional<std::string> previousResponseId;
};

/**
 * Reserve a decode-side destination KV slot for prefill-first disaggregation.
 *
 * Mirrors LLMPipeline::resolveSession slot acquisition (response-id lookup,
 * prefix-cache lookup, or new session allocation) but does not trim prompts —
 * prefill owns the full token sequence.
 *
 * On success, the session is left in-flight; the caller must release it when
 * the disaggregated request completes or is cancelled.
 */
void resolveDecodeDestinationSlot(
    SessionManager& sessionManager, const ResolveInput& input,
    trantor::EventLoop* eventLoop,
    std::function<void(DecodeDestinationSlot)> onResolved,
    std::function<void(std::string_view)> onError,
    std::function<void()> cancelFn = nullptr);

}  // namespace tt::services::decode_slot_reservation
