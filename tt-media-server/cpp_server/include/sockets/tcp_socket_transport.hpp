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

#include "sockets/i_socket_transport.hpp"
#include "sockets/socket_transport_state.hpp"
#include "utils/scoped_fd.hpp"

namespace tt::sockets {

/**
 * @brief TCP socket transport using raw POSIX sockets.
 *
 * Original transport implementation — length-prefixed framing over a single
 * TCP connection with keepalive and automatic reconnect.
 */
class TcpSocketTransport : public ISocketTransport,
                           protected SocketTransportState {
 public:
  TcpSocketTransport() = default;
  TcpSocketTransport(const TcpSocketTransport&) = delete;
  TcpSocketTransport& operator=(const TcpSocketTransport&) = delete;
  TcpSocketTransport(TcpSocketTransport&&) = delete;
  TcpSocketTransport& operator=(TcpSocketTransport&&) = delete;
  ~TcpSocketTransport() override;

  bool initializeAsServer(uint16_t port) override;
  bool initializeAsClient(const std::string& host, uint16_t port) override;

  void start() override;
  void stop() override;

  bool isConnected() const override;
  std::string getStatus() const override;

  bool sendRawData(const std::vector<uint8_t>& data) override;
  std::vector<uint8_t> receiveRawData() override;

  void setConnectionLostCallback(std::function<void()> callback) override;
  void setReconnectBackoff(uint32_t initialDelayMs,
                           uint32_t maxDelayMs) override;

 private:
  enum class ReceiveResult { COMPLETE, NO_DATA, DISCONNECTED };

  void serverLoop();
  void clientLoop();
  bool sendAll(int fd, const void* buffer, size_t size);
  ReceiveResult receiveExact(int fd, uint8_t* buffer, size_t size,
                             int maxRetries, bool returnIfNoInitialData);

  std::string host_;
  uint16_t port_;

  tt::utils::ScopedFd serverSocket_;
  tt::utils::ScopedFd clientSocket_;
  std::atomic<int> peerSocket_{-1};

  std::thread connectionThread_;

  mutable std::mutex socketMutex_;
};

}  // namespace tt::sockets
