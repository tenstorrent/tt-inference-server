// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/socket_manager.hpp"

#include <cstring>

#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
constexpr auto MESSAGE_LOOP_SLEEP = std::chrono::milliseconds(10);
}

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
  transport = createSocketTransport();
  applyPendingSettings();
  return transport->initializeAsServer(port);
}

bool SocketManager::initializeAsClient(const std::string& host, uint16_t port) {
  transport = createSocketTransport();
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
    try {
      auto data = transport->receiveRawData();
      if (!data.empty()) {
        handleIncomingMessage(data);
      }
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[SocketManager] Message loop error: {}", e.what());
    }

    std::this_thread::sleep_for(MESSAGE_LOOP_SLEEP);
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
