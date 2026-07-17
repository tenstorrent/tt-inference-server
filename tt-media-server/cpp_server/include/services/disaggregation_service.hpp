// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <trantor/net/EventLoopThread.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config/types.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/sentinel_values.hpp"
#include "sockets/socket_messages.hpp"
#include "utils/concurrent_map.hpp"

namespace tt::sockets {

class InterServerService;
}  // namespace tt::sockets

namespace tt::dynamo {
class EtcdClient;
}

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

  /** Legacy / gateway path: run prefill for a PrefillRequestMessage. */
  void handlePrefillRequest(
      const tt::sockets::PrefillRequestMessage& message,
      std::function<void(const tt::sockets::PrefillResultMessage&)> callback);

  /**
   * Prefill-first path for Dynamo (DYNAMO_ROUTING=1): discover decode via
   * etcd, reserve a decode slot over ZMQ, run one prefill token, and return a
   * PrefillResultMessage for disaggregated_params.
   */
  void handlePrefillFirstRequest(
      LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
      std::function<void(const tt::sockets::PrefillResultMessage&)> callback);

  /** Prefill-first path: etcd discovery + ZMQ slot reservation, then one
   * prefill token. Requires DYNAMO_ROUTING=1. */
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
    std::optional<
        std::function<void(const tt::sockets::PrefillResultMessage&)>>
        resultCallback;
    std::string decodeInstanceId;
  };

  struct DecodePeer {
    std::string instanceIdHex;
    uint64_t instanceId = 0;
    std::string tcpAddress;
  };

  void setupSocketHandlers();
  void handleSlotReservationRequest(
      const tt::sockets::SlotReservationRequestMessage& message);
  void handleSlotReservationResponse(
      const tt::sockets::SlotReservationResponseMessage& message);
  void launchPrefillWork(PrefillWorkContext work,
                         std::function<void(const LLMStreamChunk&, bool)> onChunk);
  void failPrefillFirstPending(uint32_t taskId, std::string_view errorText);

  void enqueuePrefillFirst(
      LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
      StreamCallback streamCallback,
      std::optional<
          std::function<void(const tt::sockets::PrefillResultMessage&)>>
          resultCallback);
  void reserveDecodeSlotViaSocket(
      uint32_t taskId, const std::vector<uint64_t>& registrationHashes,
      const LLMRequest& request);
  std::vector<DecodePeer> discoverDecodePeers() const;
  std::optional<DecodePeer> selectDecodePeer(
      const std::vector<DecodePeer>& peers) const;

  void applySlotReservationAndLaunch(
      PrefillFirstPending pending,
      const tt::sockets::SlotReservationResponseMessage& result);

  tt::config::LLMMode mode;
  std::shared_ptr<LLMService> llmService;
  std::shared_ptr<sockets::InterServerService> socketService;
  std::shared_ptr<SessionManager> sessionManager;
  trantor::EventLoopThread eventLoopThread;
  utils::ConcurrentMap<uint32_t, StreamCallback> streamCallbacks;
  utils::ConcurrentMap<uint32_t, PrefillFirstPending> pendingSlotReservations;

  std::unique_ptr<tt::dynamo::EtcdClient> etcdClient;
};

}  // namespace tt::services
