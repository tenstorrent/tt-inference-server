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
  if (reconnectBackoffSet_) {
    transport_->setReconnectBackoff(reconnectInitialDelayMs_,
                                    reconnectMaxDelayMs_);
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
  messageThread_ = std::thread(&SocketManager::messageLoop, this);
}

void SocketManager::stop() {
  if (!running_) {
    return;
  }

  running_ = false;

  if (messageThread_.joinable()) {
    messageThread_.join();
  }

  if (transport_) {
    transport_->stop();
  }

  TT_LOG_INFO("[SocketManager] Stopped");
}

void SocketManager::messageLoop() {
  while (running_) {
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
    std::string serialized(data.begin(), data.end());
    std::istringstream iss(serialized);

    cereal::BinaryInputArchive archive(iss);
    std::string messageType;
    archive(messageType);

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
    const std::string& messageType) const {
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

void SocketManager::setReconnectBackoff(uint32_t initialDelayMs,
                                        uint32_t maxDelayMs) {
  if (transport_) {
    transport_->setReconnectBackoff(initialDelayMs, maxDelayMs);
  } else {
    reconnectBackoffSet_ = true;
    reconnectInitialDelayMs_ = initialDelayMs;
    reconnectMaxDelayMs_ = maxDelayMs;
  }
}

}  // namespace tt::sockets
