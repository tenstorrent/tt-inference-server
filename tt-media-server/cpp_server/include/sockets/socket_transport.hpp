// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "utils/scoped_fd.hpp"

namespace tt::sockets {

/**
 * @brief Low-level socket transport for inter-server communication
 *
 * Handles socket lifecycle, connection management, and raw byte I/O.
 * Supports both server (listening) and client (connecting) modes.
 * Uses length-prefixed framing for message boundaries.
 */
class SocketTransport {
 public:
  enum class Mode {
    SERVER,  // Listen for incoming connections
    CLIENT   // Connect to remote server
  };

  SocketTransport() = default;
  SocketTransport(const SocketTransport&) = delete;
  SocketTransport& operator=(const SocketTransport&) = delete;
  SocketTransport(SocketTransport&&) = delete;
  SocketTransport& operator=(SocketTransport&&) = delete;
  ~SocketTransport();

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
   * @brief Start the transport (begins listening/connecting)
   */
  void start();

  /**
   * @brief Stop the transport
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
   * @brief Send raw data with length-prefix framing
   * @param data Bytes to send
   * @return true if successful
   */
  bool sendRawData(const std::vector<uint8_t>& data);

  /**
   * @brief Receive raw data with length-prefix framing
   * @return Received bytes, empty if no data or error
   */
  std::vector<uint8_t> receiveRawData();

  /**
   * @brief Set callback for connection lost events
   * @param callback Function to call when connection is lost
   */
  void setConnectionLostCallback(std::function<void()> callback);

  /**
   * @brief Set callback fired when a TCP connection is established.
   *
   * SERVER mode: each accept. CLIENT mode: each (re)connect.
   */
  void setConnectionEstablishedCallback(std::function<void()> callback);

  /**
   * @brief Configure client-mode reconnect backoff (defaults: 100ms/5000ms).
   * Delay doubles per failed attempt up to max, resets on success.
   * Must be called before start().
   */
  void setReconnectBackoff(uint32_t initial_delay_ms, uint32_t max_delay_ms);

 private:
  void serverLoop();
  void clientLoop();

  Mode mode_;
  std::string host_;
  uint16_t port_;

  tt::utils::ScopedFd serverSocket_;
  tt::utils::ScopedFd clientSocket_;
  int peerSocket_ = -1;  // Non-owning view of active connection FD

  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};

  std::thread connectionThread_;

  mutable std::mutex sendMutex_;

  std::function<void()> connectionLostCallback_;
  std::function<void()> connectionEstablishedCallback_;

  uint32_t reconnectInitialDelayMs_ = 100;
  uint32_t reconnectMaxDelayMs_ = 5000;
};

}  // namespace tt::sockets
