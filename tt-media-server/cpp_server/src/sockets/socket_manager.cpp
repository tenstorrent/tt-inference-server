// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/socket_manager.hpp"

#include <cstring>

#include "sockets/zmq_socket_transport.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
// Idle backoff only — applied when no message is pending. A burst of concurrent
// messages is drained in a single pass (see messageLoop), so this no longer
// serializes requests at the poll interval the way the old 10ms did.
constexpr auto MESSAGE_LOOP_IDLE_WAIT = std::chrono::milliseconds(1);
}  // namespace

SocketManager::~SocketManager() { stop(); }

void SocketManager::applyPendingSettings() {
  if (!transport) return;
  if (pendingConnectionLostCallback) {
    transport->setConnectionLostCallback(
        std::move(pendingConnectionLostCallback));
  }
  if (pendingConnectionEstablishedCallback) {
    transport->setConnectionEstablishedCallback(
        std::move(pendingConnectionEstablishedCallback));
  }
  if (reconnectBackoffSet) {
    transport->setReconnectBackoff(reconnectInitialDelay, reconnectMaxDelay);
  }
}

bool SocketManager::initializeAsServer(uint16_t port) {
  transport = std::make_unique<ZmqSocketTransport>();
  applyPendingSettings();
  return transport->initializeAsServer(port);
}

bool SocketManager::initializeAsClient(const std::string& host, uint16_t port) {
  transport = std::make_unique<ZmqSocketTransport>();
  applyPendingSettings();
  return transport->initializeAsClient(host, port);
}

void SocketManager::start() {
  if (running) {
    return;
  }

  running = true;
  transport->start();
  messageThread = std::jthread(
      [this](std::stop_token stopToken) { messageLoop(stopToken); });
}

void SocketManager::stop() {
  if (!running) {
    return;
  }

  running = false;

  if (messageThread.joinable()) {
    messageThread.request_stop();
    messageThread.join();
  }

  if (transport) {
    transport->stop();
  }

  TT_LOG_INFO("[SocketManager] Stopped");
}

void SocketManager::messageLoop(std::stop_token stopToken) {
  while (running && !stopToken.stop_requested()) {
    bool drainedAny = false;
    try {
      // Drain every message the transport currently has buffered in one pass.
      // Previously we handled a single message per iteration and then slept, so
      // a burst of N concurrent messages was admitted at the poll interval
      // (N * 10ms) — that was the dominant decode->prefill "socket" latency,
      // growing ~10ms per concurrent request. Draining here collapses the burst
      // into a single pass so all queued requests are admitted immediately.
      while (running && !stopToken.stop_requested()) {
        auto data = transport->receiveRawData();
        if (data.empty()) {
          break;
        }
        handleIncomingMessage(data);
        drainedAny = true;
      }
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[SocketManager] Message loop error: {}", e.what());
    }

    // Back off only when idle; kept short (1ms) so first-message latency after
    // an idle period stays low instead of the old fixed 10ms poll.
    if (!drainedAny) {
      std::this_thread::sleep_for(MESSAGE_LOOP_IDLE_WAIT);
    }
  }
}

void SocketManager::handleIncomingMessage(const std::vector<uint8_t>& data) {
  try {
    std::string messageType = wire::readMessageType(data);
    auto handler = getHandler(messageType);
    if (!handler) {
      TT_LOG_DEBUG("[SocketManager] No handler for message type: {}",
                   messageType);
      return;
    }

    handler(data);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SocketManager] Message handling error: {}", e.what());
  }
}

std::function<void(const std::vector<uint8_t>&)> SocketManager::getHandler(
    std::string_view messageType) const {
  std::lock_guard<std::mutex> lock(handlersMutex);
  auto it = handlers.find(messageType);
  if (it == handlers.end()) {
    return {};
  }
  return it->second;
}

bool SocketManager::isConnected() const {
  return transport && transport->isConnected();
}

std::string SocketManager::getStatus() const {
  return transport ? transport->getStatus() : "uninitialized";
}

void SocketManager::setConnectionLostCallback(std::function<void()> callback) {
  if (transport) {
    transport->setConnectionLostCallback(std::move(callback));
  } else {
    pendingConnectionLostCallback = std::move(callback);
  }
}

void SocketManager::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  if (transport) {
    transport->setConnectionEstablishedCallback(std::move(callback));
  } else {
    pendingConnectionEstablishedCallback = std::move(callback);
  }
}

void SocketManager::setReconnectBackoff(std::chrono::milliseconds initialDelay,
                                        std::chrono::milliseconds maxDelay) {
  if (transport) {
    transport->setReconnectBackoff(initialDelay, maxDelay);
  } else {
    reconnectBackoffSet = true;
    reconnectInitialDelay = initialDelay;
    reconnectMaxDelay = maxDelay;
  }
}

}  // namespace tt::sockets
