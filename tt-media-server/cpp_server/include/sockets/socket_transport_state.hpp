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
      : reconnectInitialDelay_(reconnectInitialDelay),
        reconnectMaxDelay_(reconnectMaxDelay) {}

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

  void setConnectionEstablishedCallbackCommon(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    connectionEstablishedCallback_ = std::move(callback);
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

  void notifyConnectionEstablished() {
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(callbackMutex_);
      callback = connectionEstablishedCallback_;
    }
    if (callback) {
      callback();
    }
  }

  void setReconnectBackoffCommon(std::chrono::milliseconds initialDelay,
                                 std::chrono::milliseconds maxDelay) {
    reconnectInitialDelay_ = initialDelay;
    reconnectMaxDelay_ = maxDelay;
  }

  Mode mode_{Mode::CLIENT};
  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};
  std::chrono::milliseconds reconnectInitialDelay_;
  std::chrono::milliseconds reconnectMaxDelay_;

 private:
  mutable std::mutex callbackMutex_;
  std::function<void()> connectionLostCallback_;
  std::function<void()> connectionEstablishedCallback_;
};

}  // namespace tt::sockets
