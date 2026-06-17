// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

namespace tt::sockets {

/**
 * @brief Shared state for concrete socket transports.
 */
class SocketTransportState {
 protected:
  enum class Mode { SERVER, CLIENT };

  SocketTransportState(std::chrono::milliseconds reconnectInitialDelay =
                           std::chrono::milliseconds(100),
                       std::chrono::milliseconds reconnectMaxDelay =
                           std::chrono::milliseconds(5000))
      : reconnectInitialDelay(reconnectInitialDelay),
        reconnectMaxDelay(reconnectMaxDelay) {}

  bool isConnectedState() const { return connected; }

  std::string getStatusString(bool connected) const {
    if (!running) {
      return "stopped";
    }
    if (connected) {
      return mode == Mode::SERVER ? "server:connected" : "client:connected";
    }
    return mode == Mode::SERVER ? "server:waiting" : "client:connecting";
  }

  const char* modeName() const {
    return mode == Mode::SERVER ? "server" : "client";
  }

  void markDisconnected() { connected = false; }

  void setConnectionLostCallbackCommon(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex);
    connectionLostCallback = std::move(callback);
  }

  void setConnectionEstablishedCallbackCommon(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex);
    connectionEstablishedCallback = std::move(callback);
  }

  void notifyConnectionLost() {
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(callbackMutex);
      callback = connectionLostCallback;
    }
    if (callback) {
      callback();
    }
  }

  void notifyConnectionEstablished() {
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(callbackMutex);
      callback = connectionEstablishedCallback;
    }
    if (callback) {
      callback();
    }
  }

  void setReconnectBackoffCommon(std::chrono::milliseconds initialDelay,
                                 std::chrono::milliseconds maxDelay) {
    reconnectInitialDelay = initialDelay;
    reconnectMaxDelay = maxDelay;
  }

  Mode mode{Mode::CLIENT};
  std::atomic<bool> running{false};
  std::atomic<bool> connected{false};
  std::chrono::milliseconds reconnectInitialDelay;
  std::chrono::milliseconds reconnectMaxDelay;

 private:
  mutable std::mutex callbackMutex;
  std::function<void()> connectionLostCallback;
  std::function<void()> connectionEstablishedCallback;
};

}  // namespace tt::sockets
