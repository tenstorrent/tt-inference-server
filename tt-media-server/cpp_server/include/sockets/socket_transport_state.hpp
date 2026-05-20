// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

namespace tt::sockets {

/**
 * @brief Shared state for concrete socket transports.
 *
 * The byte-level transport logic stays in TcpSocketTransport/ZmqSocketTransport;
 * this helper only owns common lifecycle/status/callback state.
 */
class SocketTransportState {
 protected:
  enum class Mode { SERVER, CLIENT };

  SocketTransportState(uint32_t reconnectInitialDelayMs = 100,
                       uint32_t reconnectMaxDelayMs = 5000)
      : reconnectInitialDelayMs_(reconnectInitialDelayMs),
        reconnectMaxDelayMs_(reconnectMaxDelayMs) {}

  bool isConnectedState() const { return connected_; }

  std::string getStatusString() const {
    if (!running_) {
      return "stopped";
    }
    if (connected_) {
      return mode_ == Mode::SERVER ? "server:connected" : "client:connected";
    }
    return mode_ == Mode::SERVER ? "server:waiting" : "client:connecting";
  }

  const char* modeName() const {
    return mode_ == Mode::SERVER ? "server" : "client";
  }

  void markDisconnected() { connected_ = false; }

  void setConnectionLostCallbackCommon(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    connectionLostCallback_ = std::move(callback);
  }

  void notifyConnectionLost() {
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(callbackMutex_);
      callback = connectionLostCallback_;
    }
    if (callback) {
      callback();
    }
  }

  void setReconnectBackoffCommon(uint32_t initialDelayMs,
                                 uint32_t maxDelayMs) {
    reconnectInitialDelayMs_ = initialDelayMs;
    reconnectMaxDelayMs_ = maxDelayMs;
  }

  Mode mode_{Mode::CLIENT};
  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};
  uint32_t reconnectInitialDelayMs_;
  uint32_t reconnectMaxDelayMs_;

 private:
  mutable std::mutex callbackMutex_;
  std::function<void()> connectionLostCallback_;
};

}  // namespace tt::sockets
