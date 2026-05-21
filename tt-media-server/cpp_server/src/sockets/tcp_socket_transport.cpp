// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/tcp_socket_transport.hpp"

#include <errno.h>
#include <fcntl.h>
#include <netinet/tcp.h>

#include <algorithm>
#include <cstring>

#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
constexpr int KEEPALIVE_IDLE_SECONDS = 10;
constexpr int KEEPALIVE_INTERVAL_SECONDS = 5;
constexpr int KEEPALIVE_MAX_PROBES = 3;
constexpr int MAX_SEND_RETRIES = 100;
constexpr int MAX_HEADER_RETRIES = 100;
constexpr int MAX_PAYLOAD_RETRIES = 1000;
constexpr uint32_t MAX_MESSAGE_SIZE_BYTES = 1024 * 1024;
constexpr auto RETRY_SLEEP = std::chrono::milliseconds(1);
constexpr auto CONNECTION_POLL_INTERVAL = std::chrono::milliseconds(100);

bool wouldBlock() { return errno == EAGAIN || errno == EWOULDBLOCK; }

void setSocketKeepAlive(int socketFd) {
  int enable = 1;
  if (setsockopt(socketFd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) <
      0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set SO_KEEPALIVE: {}",
                strerror(errno));
  }

  int idle = KEEPALIVE_IDLE_SECONDS;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle)) <
      0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set TCP_KEEPIDLE: {}",
                strerror(errno));
  }

  int interval = KEEPALIVE_INTERVAL_SECONDS;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPINTVL, &interval,
                 sizeof(interval)) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set TCP_KEEPINTVL: {}",
                strerror(errno));
  }

  int maxProbes = KEEPALIVE_MAX_PROBES;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPCNT, &maxProbes,
                 sizeof(maxProbes)) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set TCP_KEEPCNT: {}",
                strerror(errno));
  }
}

void setNonBlocking(int socketFd) {
  int flags = fcntl(socketFd, F_GETFL, 0);
  if (flags < 0 || fcntl(socketFd, F_SETFL, flags | O_NONBLOCK) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set non-blocking: {}",
                strerror(errno));
  }
}

void setCloseOnExec(int socketFd) {
  int flags = fcntl(socketFd, F_GETFD, 0);
  if (flags < 0 || fcntl(socketFd, F_SETFD, flags | FD_CLOEXEC) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set FD_CLOEXEC: {}",
                strerror(errno));
  }
}

tt::utils::ScopedFd createTcpSocket() {
  tt::utils::ScopedFd socketFd(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (!socketFd) {
    socketFd.reset(socket(AF_INET, SOCK_STREAM, 0));
    if (socketFd) {
      setCloseOnExec(socketFd.get());
    }
  }
  return socketFd;
}

tt::utils::ScopedFd acceptClient(int serverSocket,
                                 struct sockaddr_in* clientAddr,
                                 socklen_t* clientLen) {
  tt::utils::ScopedFd accepted(
      accept4(serverSocket, reinterpret_cast<struct sockaddr*>(clientAddr),
              clientLen, SOCK_CLOEXEC));
  if (!accepted && (errno == ENOSYS || errno == EINVAL)) {
    accepted.reset(accept(serverSocket,
                          reinterpret_cast<struct sockaddr*>(clientAddr),
                          clientLen));
    if (accepted) {
      setCloseOnExec(accepted.get());
    }
  }
  return accepted;
}

void configureSocket(int socketFd) {
  setNonBlocking(socketFd);
  setSocketKeepAlive(socketFd);
}
}  // namespace

TcpSocketTransport::~TcpSocketTransport() { stop(); }

bool TcpSocketTransport::initializeAsServer(uint16_t port) {
  mode_ = Mode::SERVER;
  port_ = port;

  serverSocket_ = createTcpSocket();
  if (!serverSocket_) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to create server socket: {}",
                 strerror(errno));
    return false;
  }

  int opt = 1;
  if (setsockopt(serverSocket_.get(), SOL_SOCKET, SO_REUSEADDR, &opt,
                 sizeof(opt)) < 0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to set SO_REUSEADDR: {}",
                 strerror(errno));
    serverSocket_.reset();
    return false;
  }

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port_);

  if (bind(serverSocket_.get(), (struct sockaddr*)&address, sizeof(address)) <
      0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to bind to port {}: {}", port_,
                 strerror(errno));
    serverSocket_.reset();
    return false;
  }

  if (listen(serverSocket_.get(), 1) < 0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to listen: {}", strerror(errno));
    serverSocket_.reset();
    return false;
  }

  TT_LOG_INFO("[TcpSocketTransport] Server initialized on port {}", port_);
  return true;
}

bool TcpSocketTransport::initializeAsClient(const std::string& host,
                                            uint16_t port) {
  mode_ = Mode::CLIENT;
  host_ = host;
  port_ = port;

  TT_LOG_INFO("[TcpSocketTransport] Client initialized to connect to {}:{}",
              host_, port_);
  return true;
}

void TcpSocketTransport::start() {
  if (running_) {
    return;
  }

  running_ = true;

  if (mode_ == Mode::SERVER) {
    connectionThread_ = std::thread(&TcpSocketTransport::serverLoop, this);
  } else {
    connectionThread_ = std::thread(&TcpSocketTransport::clientLoop, this);
  }
}

void TcpSocketTransport::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  connected_ = false;

  {
    std::lock_guard<std::mutex> lock(socketMutex_);
    int fd = peerSocket_.load(std::memory_order_acquire);
    if (fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
    }
    peerSocket_.store(-1, std::memory_order_release);
  }

  if (serverSocket_) {
    ::shutdown(serverSocket_.get(), SHUT_RDWR);
  }

  if (connectionThread_.joinable()) {
    connectionThread_.join();
  }

  {
    std::lock_guard<std::mutex> lock(socketMutex_);
    clientSocket_.reset();
    serverSocket_.reset();
  }

  TT_LOG_INFO("[TcpSocketTransport] Stopped.");
}

void TcpSocketTransport::serverLoop() {
  while (running_) {
    TT_LOG_INFO("[TcpSocketTransport] Waiting for client connection...");

    struct sockaddr_in clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    tt::utils::ScopedFd accepted =
        acceptClient(serverSocket_.get(), &clientAddr, &clientLen);
    if (!accepted) {
      if (running_) {
        TT_LOG_ERROR("[TcpSocketTransport] Accept failed: {}", strerror(errno));
      }
      break;
    }

    configureSocket(accepted.get());

    peerSocket_.store(accepted.get(), std::memory_order_release);
    connected_ = true;

    TT_LOG_INFO("[TcpSocketTransport] Client connected from {}:{}",
                inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));

    while (running_ && connected_) {
      std::this_thread::sleep_for(CONNECTION_POLL_INTERVAL);
    }

    {
      std::lock_guard<std::mutex> lock(socketMutex_);
      peerSocket_.store(-1, std::memory_order_release);
      accepted.reset();
    }
    markDisconnected();
    notifyConnectionLost();

    TT_LOG_INFO("[TcpSocketTransport] Client disconnected");
  }
}

void TcpSocketTransport::clientLoop() {
  uint32_t delayMs = reconnectInitialDelayMs_;
  auto backoff = [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    delayMs = std::min(delayMs * 2, reconnectMaxDelayMs_);
  };

  while (running_) {
    clientSocket_ = createTcpSocket();
    if (!clientSocket_) {
      TT_LOG_ERROR("[TcpSocketTransport] Failed to create client socket: {}",
                   strerror(errno));
      backoff();
      continue;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &serverAddr.sin_addr) <= 0) {
      TT_LOG_ERROR("[TcpSocketTransport] Invalid address: {}", host_);
      clientSocket_.reset();
      backoff();
      continue;
    }

    TT_LOG_INFO(
        "[TcpSocketTransport] Attempting to connect to {}:{} (backoff {}ms)",
        host_, port_, delayMs);

    if (connect(clientSocket_.get(), (struct sockaddr*)&serverAddr,
                sizeof(serverAddr)) < 0) {
      TT_LOG_ERROR("[TcpSocketTransport] Connection failed: {}",
                   strerror(errno));
      clientSocket_.reset();
      backoff();
      continue;
    }

    configureSocket(clientSocket_.get());

    peerSocket_.store(clientSocket_.get(), std::memory_order_release);
    connected_ = true;
    delayMs = reconnectInitialDelayMs_;  // reset on success

    TT_LOG_INFO("[TcpSocketTransport] Connected to server");

    while (running_ && connected_) {
      std::this_thread::sleep_for(CONNECTION_POLL_INTERVAL);
    }

    {
      std::lock_guard<std::mutex> lock(socketMutex_);
      peerSocket_.store(-1, std::memory_order_release);
      clientSocket_.reset();
    }
    markDisconnected();
    notifyConnectionLost();

    TT_LOG_INFO("[TcpSocketTransport] Disconnected from server");
  }
}

bool TcpSocketTransport::sendRawData(const std::vector<uint8_t>& data) {
  std::lock_guard<std::mutex> lock(socketMutex_);
  if (!connected_) return false;

  int fd = peerSocket_.load(std::memory_order_acquire);
  if (fd < 0) return false;

  uint32_t size = static_cast<uint32_t>(data.size());
  uint32_t netSize = htonl(size);

  if (!sendAll(fd, &netSize, sizeof(netSize))) return false;
  if (!sendAll(fd, data.data(), data.size())) return false;

  return true;
}

bool TcpSocketTransport::sendAll(int fd, const void* buffer, size_t size) {
  size_t sent = 0;
  int retries = 0;
  const auto* data = static_cast<const uint8_t*>(buffer);

  while (sent < size) {
    ssize_t n = send(fd, data + sent, size - sent, MSG_NOSIGNAL);
    if (n > 0) {
      sent += static_cast<size_t>(n);
      retries = 0;
      continue;
    }

    if (n == 0 || !wouldBlock()) {
      markDisconnected();
      return false;
    }

    if (++retries > MAX_SEND_RETRIES) {
      markDisconnected();
      return false;
    }
    std::this_thread::sleep_for(RETRY_SLEEP);
  }

  return true;
}

std::vector<uint8_t> TcpSocketTransport::receiveRawData() {
  std::lock_guard<std::mutex> lock(socketMutex_);
  if (!connected_) return {};

  int fd = peerSocket_.load(std::memory_order_acquire);
  if (fd < 0) return {};

  uint32_t netSize = 0;
  auto headerStatus =
      receiveExact(fd, reinterpret_cast<uint8_t*>(&netSize), sizeof(netSize),
                   MAX_HEADER_RETRIES, /*returnIfNoInitialData=*/true);
  if (headerStatus == ReceiveResult::NO_DATA) {
    return {};
  }
  if (headerStatus == ReceiveResult::DISCONNECTED) {
    markDisconnected();
    return {};
  }

  uint32_t size = ntohl(netSize);
  if (size == 0 || size > MAX_MESSAGE_SIZE_BYTES) {
    markDisconnected();
    return {};
  }

  std::vector<uint8_t> data(size);
  auto payloadStatus =
      receiveExact(fd, data.data(), data.size(), MAX_PAYLOAD_RETRIES,
                   /*returnIfNoInitialData=*/false);
  if (payloadStatus != ReceiveResult::COMPLETE) {
    markDisconnected();
    return {};
  }

  return data;
}

TcpSocketTransport::ReceiveResult TcpSocketTransport::receiveExact(
    int fd, uint8_t* buffer, size_t size, int maxRetries,
    bool returnIfNoInitialData) {
  size_t receivedTotal = 0;
  int retries = 0;

  while (receivedTotal < size) {
    const int flags =
        receivedTotal == 0 && returnIfNoInitialData ? MSG_DONTWAIT : 0;
    ssize_t received =
        recv(fd, buffer + receivedTotal, size - receivedTotal, flags);

    if (received > 0) {
      receivedTotal += static_cast<size_t>(received);
      retries = 0;
      continue;
    }

    if (received == 0 || !wouldBlock()) {
      return ReceiveResult::DISCONNECTED;
    }

    if (receivedTotal == 0 && returnIfNoInitialData) {
      return ReceiveResult::NO_DATA;
    }

    if (++retries > maxRetries) {
      return ReceiveResult::DISCONNECTED;
    }
    std::this_thread::sleep_for(RETRY_SLEEP);
  }

  return ReceiveResult::COMPLETE;
}

bool TcpSocketTransport::isConnected() const { return isConnectedState(); }

std::string TcpSocketTransport::getStatus() const { return getStatusString(); }

void TcpSocketTransport::setConnectionLostCallback(
    std::function<void()> callback) {
  setConnectionLostCallbackCommon(std::move(callback));
}

void TcpSocketTransport::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  setConnectionEstablishedCallbackCommon(std::move(callback));
}

void TcpSocketTransport::setReconnectBackoff(uint32_t initialDelayMs,
                                             uint32_t maxDelayMs) {
  setReconnectBackoffCommon(initialDelayMs, maxDelayMs);
}

}  // namespace tt::sockets
