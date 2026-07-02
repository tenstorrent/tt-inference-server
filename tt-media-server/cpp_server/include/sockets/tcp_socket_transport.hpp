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
#include <span>
#include <stop_token>
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

  bool sendRawData(std::span<const uint8_t> data) override;
  std::vector<uint8_t> receiveRawData() override;
  ReceiveResult tryReceiveMessage() override;

  void setConnectionLostCallback(std::function<void()> callback) override;
  void setConnectionEstablishedCallback(
      std::function<void()> callback) override;
  void setReconnectBackoff(std::chrono::milliseconds initialDelay,
                           std::chrono::milliseconds maxDelay) override;

 private:
  enum class ReadResult { COMPLETE, NO_DATA, DISCONNECTED };

  void serverLoop(std::stop_token stopToken);
  void clientLoop(std::stop_token stopToken);
  bool sendAll(int fd, const void* buffer, size_t size);
  ReadResult receiveExact(int fd, uint8_t* buffer, size_t size, int maxRetries,
                          bool returnIfNoInitialData);

  std::string host;
  uint16_t port;

  tt::utils::ScopedFd serverSocket;
  tt::utils::ScopedFd clientSocket;
  std::atomic<int> peerSocket{-1};

  std::jthread connectionThread;

  mutable std::mutex socketMutex;
};

}  // namespace tt::sockets
