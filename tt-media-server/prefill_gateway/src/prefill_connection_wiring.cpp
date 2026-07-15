// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_connection_wiring.hpp"

#include <string>

#include "gateway/dispatcher.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/zmq_prefill_router.hpp"
#include "sockets/socket_messages.hpp"
#include "utils/logger.hpp"

namespace tt::gateway {
namespace {

enum class RegistrationLogReason {
  NONE,
  NEW,
  RECOVERED,
  CAPACITY_CHANGED,
};

RegistrationLogReason registrationLogReason(
    const PrefillRegistry& registry,
    const tt::sockets::PrefillRegistrationMessage& msg) {
  for (const auto& prefill : registry.snapshot()) {
    if (prefill.serverId != msg.serverId) {
      continue;
    }
    if (!prefill.healthy) {
      return RegistrationLogReason::RECOVERED;
    }
    if (prefill.maxInFlight != msg.maxInFlight) {
      return RegistrationLogReason::CAPACITY_CHANGED;
    }
    return RegistrationLogReason::NONE;
  }
  return RegistrationLogReason::NEW;
}

void logPrefillRegistrationIfNeeded(
    RegistrationLogReason reason,
    const tt::sockets::PrefillRegistrationMessage& msg) {
  switch (reason) {
    case RegistrationLogReason::NEW:
      TT_LOG_INFO("[Gateway] Prefill registered: id='{}' max_in_flight={}",
                  msg.serverId, msg.maxInFlight);
      break;
    case RegistrationLogReason::RECOVERED:
      TT_LOG_INFO("[Gateway] Prefill recovered: id='{}' max_in_flight={}",
                  msg.serverId, msg.maxInFlight);
      break;
    case RegistrationLogReason::CAPACITY_CHANGED:
      TT_LOG_INFO(
          "[Gateway] Prefill capacity changed: id='{}' max_in_flight={}",
          msg.serverId, msg.maxInFlight);
      break;
    case RegistrationLogReason::NONE:
      break;
  }
}

}  // namespace

void registerZmqPrefillHandlers(ZmqPrefillRouter& zmqPrefillRouter,
                                PrefillRegistry& registry,
                                Dispatcher& dispatcher) {
  zmqPrefillRouter.registerHandler<tt::sockets::PrefillRegistrationMessage>(
      tt::sockets::tags::PREFILL_REGISTRATION,
      [&registry, &zmqPrefillRouter](
          const ZmqPrefillRouter::PeerIdentity& peerId,
          const tt::sockets::PrefillRegistrationMessage& msg) {
        const auto logReason = registrationLogReason(registry, msg);
        zmqPrefillRouter.rememberRegistration(msg.serverId, peerId);
        registry.preRegister(msg.serverId, nullptr);
        bool ok = registry.markRegistered(msg.serverId, msg.maxInFlight);
        if (!ok) {
          TT_LOG_ERROR("[Gateway] markRegistered failed for '{}'",
                       msg.serverId);
        } else {
          logPrefillRegistrationIfNeeded(logReason, msg);
        }
      });

  zmqPrefillRouter.registerHandler<tt::sockets::PrefillResultMessage>(
      tt::sockets::tags::PREFILL_RESULT,
      [&dispatcher, &zmqPrefillRouter](
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

  zmqPrefillRouter.registerHandler<tt::sockets::SlotReservationRequestMessage>(
      tt::sockets::tags::SLOT_RESERVATION_REQUEST,
      [&dispatcher, &zmqPrefillRouter](
          const ZmqPrefillRouter::PeerIdentity& peerId,
          const tt::sockets::SlotReservationRequestMessage& msg) {
        auto serverId = zmqPrefillRouter.serverIdForPeer(peerId);
        if (!serverId.has_value()) {
          TT_LOG_WARN(
              "[Gateway] Ignoring slot reservation request from unregistered "
              "prefill");
          return;
        }
        dispatcher.onSlotReservationRequest(*serverId, msg);
      });
}

}  // namespace tt::gateway
