// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "sockets/socket_serialization.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

/**
 * @brief Singleton socket manager for inter-server communication
 *
 * Supports both server (listening) and client (connecting) modes.
 * Can serialize and send objects using Cereal library.
 */
class SocketManager {
 public:
  SocketManager() = default;
  SocketManager(const SocketManager&) = delete;
  SocketManager& operator=(const SocketManager&) = delete;
  SocketManager(SocketManager&&) = delete;
  SocketManager& operator=(SocketManager&&) = delete;
  ~SocketManager();

  /**
   * @brief Initialize as server (listening mode)
   * @param port Port to listen on
   * @return true if successful
   */
  bool initializeAsServer(uint16_t port);

  /**
   * @brief Initialize as client (connecting mode)
   * @param host Remote host to connect to
   * @param port Remote port to connect to
   * @return true if successful
   */
  bool initializeAsClient(const std::string& host, uint16_t port);

  /**
   * @brief Send serializable object to connected peer
   * @param messageType Type identifier for the message
   * @param obj Object to send
   * @return true if successful
   */
  template <typename T>
  bool sendObject(std::string_view messageType, const T& obj);

  /**
   * @brief Register handler for incoming messages of specific type
   * @param messageType Type identifier to handle
   * @param handler Function to call when message is received
   */
  template <typename T>
  void registerHandler(std::string_view messageType,
                       std::function<void(const T&)> handler);

  /**
   * @brief Start the socket manager (begins listening/connecting)
   */
  void start();

  /**
   * @brief Stop the socket manager
   */
  void stop();

  /**
   * @brief Check if connected to peer
   */
  bool isConnected() const;

  /**
   * @brief Get connection status string
   */
  std::string getStatus() const;

  /**
   * @brief Set callback for connection lost events.
   */
  void setConnectionLostCallback(std::function<void()> callback);

  /**
   * @brief Set callback for connection established/reconnected events.
   */
  void setConnectionEstablishedCallback(std::function<void()> callback);

  /**
   * @brief Configure client-mode reconnect backoff (defaults: 100ms/5000ms).
   * Must be called before start().
   */
  void setReconnectBackoff(std::chrono::milliseconds initialDelay,
                           std::chrono::milliseconds maxDelay);

 private:
  void messageLoop(std::stop_token stopToken);
  void handleIncomingMessage(const std::vector<uint8_t>& data);
  std::function<void(const std::vector<uint8_t>&)> getHandler(
      std::string_view messageType) const;

  std::unique_ptr<ISocketTransport> transport_;

  std::atomic<bool> running_{false};
  std::jthread messageThread_;

  mutable std::mutex handlersMutex_;
  std::map<std::string, std::function<void(const std::vector<uint8_t>&)>,
           std::less<>>
      handlers_;

  std::function<void()> pendingConnectionLostCallback_;
  std::function<void()> pendingConnectionEstablishedCallback_;
  bool reconnectBackoffSet_{false};
  std::chrono::milliseconds reconnectInitialDelay_{0};
  std::chrono::milliseconds reconnectMaxDelay_{0};

  void applyPendingSettings();
};

// Template implementations

template <typename T>
bool SocketManager::sendObject(std::string_view messageType, const T& obj) {
  if (!transport_) {
    return false;
  }

  try {
    std::vector<uint8_t> data = wire::serializeMessage(messageType, obj);
    return transport_->sendRawData(data);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SocketManager] Serialization error: {}", e.what());
    return false;
  }
}

template <typename T>
void SocketManager::registerHandler(std::string_view messageType,
                                    std::function<void(const T&)> handler) {
  std::lock_guard<std::mutex> lock(handlersMutex_);

  handlers_[std::string(messageType)] =
      [handler](const std::vector<uint8_t>& data) {
        try {
          T payload = wire::deserializePayload<T>(data);
          handler(payload);
        } catch (const std::exception& e) {
          TT_LOG_ERROR("[SocketManager] Deserialization error: {}", e.what());
        }
      };
}

}  // namespace tt::sockets
