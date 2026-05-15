// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/socket_manager.hpp"

#include <cstring>

#include "utils/logger.hpp"

namespace tt::sockets {

SocketManager::~SocketManager() { stop(); }

bool SocketManager::initializeAsServer(uint16_t port) {
  return transport_.initializeAsServer(port);
}

bool SocketManager::initializeAsClient(const std::string& host, uint16_t port) {
  return transport_.initializeAsClient(host, port);
}

void SocketManager::start() {
  if (running_) {
    return;
  }

  running_ = true;
  transport_.start();
  messageThread_ = std::thread(&SocketManager::messageLoop, this);
}

void SocketManager::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  transport_.stop();

  if (messageThread_.joinable()) {
    messageThread_.join();
  }

  TT_LOG_INFO("[SocketManager] Stopped");
}

void SocketManager::messageLoop() {
  while (running_) {
    if (!transport_.isConnected()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    try {
      auto data = transport_.receiveRawData();
      if (!data.empty()) {
        handleIncomingMessage(data);
      }
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[SocketManager] Message loop error: {}", e.what());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void SocketManager::handleIncomingMessage(const std::vector<uint8_t>& data) {
  try {
    std::string serialized(data.begin(), data.end());
    std::istringstream iss(serialized);

    cereal::BinaryInputArchive archive(iss);
    std::string messageType;
    archive(messageType);

    std::lock_guard<std::mutex> lock(handlersMutex_);
    auto it = handlers_.find(messageType);
    if (it != handlers_.end()) {
      it->second(data);
    } else {
      TT_LOG_DEBUG("[SocketManager] No handler for message type: {}",
                   messageType);
    }
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SocketManager] Message handling error: {}", e.what());
  }
}

bool SocketManager::isConnected() const { return transport_.isConnected(); }

std::string SocketManager::getStatus() const { return transport_.getStatus(); }

void SocketManager::setConnectionLostCallback(std::function<void()> callback) {
  transport_.setConnectionLostCallback(std::move(callback));
}

void SocketManager::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  transport_.setConnectionEstablishedCallback(std::move(callback));
}

void SocketManager::setReconnectBackoff(uint32_t initialDelayMs,
                                        uint32_t maxDelayMs) {
  transport_.setReconnectBackoff(initialDelayMs, maxDelayMs);
}

}  // namespace tt::sockets
