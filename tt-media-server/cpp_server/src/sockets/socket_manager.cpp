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
  if (!transport_) return;
  if (pendingConnectionLostCallback_) {
    transport_->setConnectionLostCallback(
        std::move(pendingConnectionLostCallback_));
  }
  if (pendingConnectionEstablishedCallback_) {
    transport_->setConnectionEstablishedCallback(
        std::move(pendingConnectionEstablishedCallback_));
  }
  if (reconnectBackoffSet_) {
    transport_->setReconnectBackoff(reconnectInitialDelay_, reconnectMaxDelay_);
  }
}

bool SocketManager::initializeAsServer(uint16_t port) {
  transport_ = createSocketTransport();
  applyPendingSettings();
  return transport_->initializeAsServer(port);
}

bool SocketManager::initializeAsClient(const std::string& host, uint16_t port) {
  transport_ = createSocketTransport();
  applyPendingSettings();
  return transport_->initializeAsClient(host, port);
}

void SocketManager::start() {
  if (running_) {
    return;
  }

  running_ = true;
  transport_->start();
  messageThread_ = std::jthread(
      [this](std::stop_token stopToken) { messageLoop(stopToken); });
}

void SocketManager::stop() {
  if (!running_) {
    return;
  }

  running_ = false;

  if (messageThread_.joinable()) {
    messageThread_.request_stop();
    messageThread_.join();
  }

  if (transport_) {
    transport_->stop();
  }

  TT_LOG_INFO("[SocketManager] Stopped");
}

void SocketManager::messageLoop(std::stop_token stopToken) {
  while (running_ && !stopToken.stop_requested()) {
    try {
      auto data = transport_->receiveRawData();
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
  std::lock_guard<std::mutex> lock(handlersMutex_);
  auto it = handlers_.find(messageType);
  if (it == handlers_.end()) {
    return {};
  }
  return it->second;
}

bool SocketManager::isConnected() const {
  return transport_ && transport_->isConnected();
}

std::string SocketManager::getStatus() const {
  return transport_ ? transport_->getStatus() : "uninitialized";
}

void SocketManager::setConnectionLostCallback(std::function<void()> callback) {
  if (transport_) {
    transport_->setConnectionLostCallback(std::move(callback));
  } else {
    pendingConnectionLostCallback_ = std::move(callback);
  }
}

void SocketManager::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  if (transport_) {
    transport_->setConnectionEstablishedCallback(std::move(callback));
  } else {
    pendingConnectionEstablishedCallback_ = std::move(callback);
  }
}

void SocketManager::setReconnectBackoff(std::chrono::milliseconds initialDelay,
                                        std::chrono::milliseconds maxDelay) {
  if (transport_) {
    transport_->setReconnectBackoff(initialDelay, maxDelay);
  } else {
    reconnectBackoffSet_ = true;
    reconnectInitialDelay_ = initialDelay;
    reconnectMaxDelay_ = maxDelay;
  }
}

}  // namespace tt::sockets
