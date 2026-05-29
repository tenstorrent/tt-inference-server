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

using namespace tt::domain::llm;
class InterServerService;
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
  void abortRequest(uint32_t taskId);

  /// Resolve a prefill-side session via prefix-cache lookup. On HIT, sets
  /// request.prefillSlotId and trims the prompt. No async allocation needed.
  void resolvePrefillSession(LLMRequest& request,
                             const std::vector<uint64_t>& routingHashes);

  void setSessionManager(std::shared_ptr<SessionManager> sm) {
    sessionManager = std::move(sm);
  }

 private:
  void setupSocketHandlers();
  void applyDeltaPrompt(LLMRequest& req, uint32_t matchedTokens);

  tt::config::LLMMode mode;
  std::shared_ptr<LLMService> llmService;
  std::shared_ptr<sockets::InterServerService> socketService;
  std::shared_ptr<SessionManager> sessionManager;
  trantor::EventLoopThread eventLoopThread;
  utils::ConcurrentMap<uint32_t, StreamCallback> streamCallbacks;
};

}  // namespace tt::services
