// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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
 * Length-prefixed framing with keepalive. Client mode auto-reconnects.
 * Server mode either:
 *   - legacy single-peer (default): accept one client, hold until disconnect;
 *   - multi-accept: when enableMultiAccept() is set before start(), every
 *     accepted FD becomes a peer transport handed to the handler and the loop
 *     keeps accepting — required so multiple prefills share one decode port.
 */
class TcpSocketTransport : public ISocketTransport,
                           protected SocketTransportState {
 public:
  /// Invoked on the accept thread with a connected peer transport.
  using AcceptHandler = sockets::ISocketTransport::AcceptHandler;

  TcpSocketTransport() = default;
  TcpSocketTransport(const TcpSocketTransport&) = delete;
  TcpSocketTransport& operator=(const TcpSocketTransport&) = delete;
  TcpSocketTransport(TcpSocketTransport&&) = delete;
  TcpSocketTransport& operator=(TcpSocketTransport&&) = delete;
  ~TcpSocketTransport() override;

  /// Wrap an already-connected peer FD (no connect/accept thread).
  static std::shared_ptr<TcpSocketTransport> fromConnectedFd(
      tt::utils::ScopedFd connectedFd);

  bool initializeAsServer(uint16_t port) override;
  bool initializeAsClient(const std::string& host, uint16_t port) override;

  /// Multi-accept mode: must be set BEFORE start(). Each accept builds a
  /// fromConnectedFd peer and invokes the handler; the listen loop continues.
  bool enableMultiAccept(AcceptHandler handler) override;

  void start() override;
  void stop() override;

  bool isConnected() const override;
  std::string getStatus() const override;

  bool sendRawData(std::span<const uint8_t> data) override;
  std::vector<uint8_t> receiveRawData() override;
  ReceiveResult tryReceiveMessage() override;

  void beginIoBudget(std::chrono::milliseconds budget) override;
  void clearIoBudget() override;

  void setConnectionLostCallback(std::function<void()> callback) override;
  void setConnectionEstablishedCallback(
      std::function<void()> callback) override;
  void setReconnectBackoff(std::chrono::milliseconds initialDelay,
                           std::chrono::milliseconds maxDelay) override;

 private:
  enum class ReadResult { COMPLETE, NO_DATA, DISCONNECTED, TIMED_OUT };

  void serverLoop(std::stop_token stopToken);
  void clientLoop(std::stop_token stopToken);
  bool sendAll(int fd, const void* buffer, size_t size);
  ReadResult receiveExact(int fd, uint8_t* buffer, size_t size, int maxRetries,
                          bool returnIfNoInitialData);
  bool isIoBudgetExpired() const;

  std::string host;
  uint16_t port = 0;

  tt::utils::ScopedFd serverSocket;
  tt::utils::ScopedFd clientSocket;
  std::atomic<int> peerSocket{-1};
  // 0 = no budget; else steady_clock epoch nanos. Set outside socketMutex.
  std::atomic<std::int64_t> ioDeadlineNs_{0};

  std::jthread connectionThread;
  AcceptHandler acceptHandler_;

  mutable std::mutex socketMutex;
};

}  // namespace tt::sockets
