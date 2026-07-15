// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <trantor/net/EventLoopThread.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "config/types.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "utils/concurrent_map.hpp"

namespace tt::sockets {

class InterServerService;
struct SlotReservationRequestMessage;
struct SlotReservationResponseMessage;
}  // namespace tt::sockets

namespace tt::services {

using namespace tt::domain::llm;

class LLMService;
class SessionManager;

class DisaggregationService {
  using StreamCallback = std::function<void(const LLMStreamChunk&, bool)>;

 public:
  DisaggregationService(
      tt::config::LLMMode mode, std::shared_ptr<LLMService> llmService,
      std::shared_ptr<sockets::InterServerService> socketService,
      std::shared_ptr<SessionManager> sessionManager = nullptr);
  ~DisaggregationService();

  void start();
  void stop();

  void handleStreamingRequest(LLMRequest& request,
                              const std::vector<uint64_t>& registrationHashes,
                              const StreamCallback& callback);

  /** Prefill-first path: reserve a decode slot, then run one prefill token. */
  void handlePrefillFirstStreamingRequest(
      LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
      const StreamCallback& callback);

  void abortRequest(uint32_t taskId);

  /// Resolve a prefill-side session via prefix-cache lookup.
  /// On HIT, sets request.prefillSlotId and trims the prompt, then calls
  /// onResolved. On MISS, allocates a new session asynchronously and calls
  /// onResolved once allocated (or onError on failure).
  void resolvePrefillSession(std::shared_ptr<LLMRequest> request,
                             const std::vector<uint64_t>& routingHashes,
                             std::function<void()> onResolved,
                             std::function<void(std::string_view)> onError);

  void setSessionManager(std::shared_ptr<SessionManager> sm) {
    sessionManager = std::move(sm);
  }

 private:
  struct PrefillWorkContext {
    std::shared_ptr<LLMRequest> request;
    std::vector<uint32_t> fullPromptTokenIds;
    uint32_t decodeSlotId = tt::domain::INVALID_SLOT_ID;
    std::optional<int> maxTokens;
    uint32_t registrationHashCount = 0;
  };

  struct PrefillFirstPending {
    PrefillWorkContext work;
    StreamCallback callback;
    std::vector<uint64_t> registrationHashes;
  };

  void setupSocketHandlers();
  void handleSlotReservationRequest(
      const tt::sockets::SlotReservationRequestMessage& message);
  void handleSlotReservationResponse(
      const tt::sockets::SlotReservationResponseMessage& message);
  void launchPrefillWork(PrefillWorkContext work,
                         std::function<void(const LLMStreamChunk&, bool)> onChunk);
  void failPrefillFirstPending(uint32_t taskId, std::string_view errorText);

  tt::config::LLMMode mode;
  std::shared_ptr<LLMService> llmService;
  std::shared_ptr<sockets::InterServerService> socketService;
  std::shared_ptr<SessionManager> sessionManager;
  trantor::EventLoopThread eventLoopThread;
  utils::ConcurrentMap<uint32_t, StreamCallback> streamCallbacks;
  utils::ConcurrentMap<uint32_t, PrefillFirstPending> pendingSlotReservations;
};

}  // namespace tt::services
