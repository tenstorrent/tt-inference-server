// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_connection_wiring.hpp"

#include <mutex>
#include <string>

#include "gateway/dispatcher.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/zmq_prefill_router.hpp"
#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "utils/logger.hpp"

namespace tt::gateway {
namespace {

struct PrefillConnectionState {
  void setServerId(const std::string& serverId) {
    std::lock_guard<std::mutex> lock(mutex);
    this->serverId = serverId;
  }

  std::string getServerId() const {
    std::lock_guard<std::mutex> lock(mutex);
    return serverId;
  }

  mutable std::mutex mutex;
  std::string serverId;
};

}  // namespace

void registerTcpPrefillHandlers(PrefillSocketManagers& prefillSms,
                                PrefillRegistry& registry,
                                Dispatcher& dispatcher) {
  for (auto& smPtr : prefillSms) {
    tt::sockets::SocketManager* sm = smPtr.get();

    // Shared between callbacks that may run on different threads. The id is
    // unknown until the first registration message.
    auto state = std::make_shared<PrefillConnectionState>();

    sm->registerHandler<tt::sockets::PrefillRegistrationMessage>(
        tt::sockets::tags::PREFILL_REGISTRATION,
        [&registry, sm,
         state](const tt::sockets::PrefillRegistrationMessage& msg) {
          TT_LOG_DEBUG("[Gateway] Prefill registered: id='{}' max_in_flight={}",
                       msg.server_id, msg.max_in_flight);
          state->setServerId(msg.server_id);
          registry.preRegister(msg.server_id, sm);
          bool ok = registry.markRegistered(msg.server_id, msg.max_in_flight);
          if (!ok) {
            TT_LOG_ERROR("[Gateway] markRegistered failed for '{}'",
                         msg.server_id);
          }
        });

    sm->registerHandler<tt::sockets::PrefillResultMessage>(
        "prefill_result",
        [&dispatcher, state](const tt::sockets::PrefillResultMessage& msg) {
          dispatcher.onPrefillResult(state->getServerId(), msg);
        });

    sm->registerHandler<tt::sockets::PrefillCacheBlocksAddedMessage>(
        tt::sockets::tags::PREFILL_CACHE_BLOCKS_ADDED,
        [&dispatcher](const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
          dispatcher.onCacheBlocksAdded(msg);
        });

    sm->registerHandler<tt::sockets::PrefillCacheBlocksEvictedMessage>(
        tt::sockets::tags::PREFILL_CACHE_BLOCKS_EVICTED,
        [&dispatcher](
            const tt::sockets::PrefillCacheBlocksEvictedMessage& msg) {
          dispatcher.onCacheBlocksEvicted(msg);
        });

    sm->setConnectionLostCallback([&registry, state]() {
      const std::string sid = state->getServerId();
      if (!sid.empty()) {
        TT_LOG_WARN("[Gateway] Prefill '{}' connection lost", sid);
        registry.markDown(sid);
      }
    });
  }
}

void registerZmqPrefillHandlers(ZmqPrefillRouter& zmqPrefillRouter,
                                PrefillRegistry& registry,
                                Dispatcher& dispatcher) {
  zmqPrefillRouter.registerHandler<tt::sockets::PrefillRegistrationMessage>(
      tt::sockets::tags::PREFILL_REGISTRATION,
      [&registry, &zmqPrefillRouter](
          const ZmqPrefillRouter::PeerIdentity& peerId,
          const tt::sockets::PrefillRegistrationMessage& msg) {
        TT_LOG_DEBUG("[Gateway] Prefill registered: id='{}' max_in_flight={}",
                     msg.server_id, msg.max_in_flight);
        zmqPrefillRouter.rememberRegistration(msg.server_id, peerId);
        registry.preRegister(msg.server_id, nullptr);
        bool ok = registry.markRegistered(msg.server_id, msg.max_in_flight);
        if (!ok) {
          TT_LOG_ERROR("[Gateway] markRegistered failed for '{}'",
                       msg.server_id);
        }
      });

  zmqPrefillRouter.registerHandler<tt::sockets::PrefillResultMessage>(
      "prefill_result", [&dispatcher, &zmqPrefillRouter](
                            const ZmqPrefillRouter::PeerIdentity& peerId,
                            const tt::sockets::PrefillResultMessage& msg) {
        auto serverId = zmqPrefillRouter.serverIdForPeer(peerId);
        if (!serverId.has_value()) {
          TT_LOG_WARN("[Gateway] Ignoring result from unregistered prefill");
          return;
        }
        dispatcher.onPrefillResult(*serverId, msg);
      });

  zmqPrefillRouter.registerHandler<tt::sockets::PrefillCacheBlocksAddedMessage>(
      tt::sockets::tags::PREFILL_CACHE_BLOCKS_ADDED,
      [&dispatcher](const ZmqPrefillRouter::PeerIdentity&,
                    const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
        dispatcher.onCacheBlocksAdded(msg);
      });

  zmqPrefillRouter
      .registerHandler<tt::sockets::PrefillCacheBlocksEvictedMessage>(
          tt::sockets::tags::PREFILL_CACHE_BLOCKS_EVICTED,
          [&dispatcher](
              const ZmqPrefillRouter::PeerIdentity&,
              const tt::sockets::PrefillCacheBlocksEvictedMessage& msg) {
            dispatcher.onCacheBlocksEvicted(msg);
          });
}

}  // namespace tt::gateway
