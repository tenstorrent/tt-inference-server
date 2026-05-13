// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "utils/logger.hpp"
#include "utils/scoped_fd.hpp"

namespace tt::sockets {

/**
 * @brief Singleton socket manager for inter-server communication
 *
 * Supports both server (listening) and client (connecting) modes.
 * Can serialize and send objects using Cereal library.
 */
class SocketManager {
 public:
  enum class Mode {
    SERVER,  // Listen for incoming connections
    CLIENT   // Connect to remote server
  };

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
   * @param message_type Type identifier for the message
   * @param obj Object to send
   * @return true if successful
   */
  template <typename T>
  bool sendObject(const std::string& messageType, const T& obj);

  /**
   * @brief Register handler for incoming messages of specific type
   * @param message_type Type identifier to handle
   * @param handler Function to call when message is received
   */
  template <typename T>
  void registerHandler(const std::string& messageType,
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
   * @param callback Function called on the internal connection thread.
   */
  void setConnectionLostCallback(std::function<void()> callback);

  /**
   * @brief Set callback fired when a connection is established (TCP level).
   *
   * SERVER mode: called each time a new client is accepted.
   * CLIENT mode: called each time a (re)connect succeeds.
   *
   * Useful for the gateway to know a socket is live before the
   * PrefillRegistrationMessage arrives. May be called from a background thread.
   */
  void setConnectionEstablishedCallback(std::function<void()> callback);

  /**
   * @brief Configure client-mode reconnect backoff.
   *
   * On each failed connect attempt the delay doubles until it reaches
   * `max_delay_ms`. The delay resets to `initial_delay_ms` on success.
   * Defaults: initial=100ms, max=5000ms.
   *
   * Must be called before start().
   */
  void setReconnectBackoff(uint32_t initial_delay_ms, uint32_t max_delay_ms);

 private:
  void serverLoop();
  void clientLoop();
  void messageLoop();
  void handleIncomingMessage(const std::vector<uint8_t>& data);
  bool sendRawData(const std::vector<uint8_t>& data);
  std::vector<uint8_t> receiveRawData();

  Mode mode_;
  std::string host_;
  uint16_t port_;

  tt::utils::ScopedFd serverSocket;
  tt::utils::ScopedFd clientSocket;
  int peerSocket = -1;  // Non-owning view of active connection FD

  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};

  std::thread server_thread_;
  std::thread message_thread_;

  mutable std::mutex handlers_mutex_;
  std::map<std::string, std::function<void(const std::vector<uint8_t>&)>>
      handlers_;

  mutable std::mutex send_mutex_;

  std::function<void()> connection_lost_callback_;
  std::function<void()> connection_established_callback_;

  uint32_t reconnect_initial_delay_ms_ = 100;
  uint32_t reconnect_max_delay_ms_ = 5000;
};

// Template implementations

template <typename T>
bool SocketManager::sendObject(const std::string& messageType, const T& obj) {
  if (!connected_) {
    return false;
  }

  try {
    std::ostringstream oss;
    {
      cereal::BinaryOutputArchive archive(oss);
      archive(messageType);
      obj.write(archive);
    }

    std::string serialized = oss.str();
    std::vector<uint8_t> data(serialized.begin(), serialized.end());

    return sendRawData(data);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SocketManager] Serialization error: {}", e.what());
    return false;
  }
}

template <typename T>
void SocketManager::registerHandler(const std::string& messageType,
                                    std::function<void(const T&)> handler) {
  std::lock_guard<std::mutex> lock(handlers_mutex_);

  handlers_[messageType] = [handler](const std::vector<uint8_t>& data) {
    try {
      std::string serialized(data.begin(), data.end());
      std::istringstream iss(serialized);

      cereal::BinaryInputArchive archive(iss);
      std::string msgType;
      archive(msgType);
      T payload = T::read(archive);

      handler(payload);
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[SocketManager] Deserialization error: {}", e.what());
    }
  };
}

}  // namespace tt::sockets
