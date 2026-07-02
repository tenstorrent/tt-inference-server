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
  mode = Mode::SERVER;
  this->port = port;

  serverSocket = createTcpSocket();
  if (!serverSocket) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to create server socket: {}",
                 strerror(errno));
    return false;
  }

  int opt = 1;
  if (setsockopt(serverSocket.get(), SOL_SOCKET, SO_REUSEADDR, &opt,
                 sizeof(opt)) < 0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to set SO_REUSEADDR: {}",
                 strerror(errno));
    serverSocket.reset();
    return false;
  }

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  if (bind(serverSocket.get(), (struct sockaddr*)&address, sizeof(address)) <
      0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to bind to port {}: {}", port,
                 strerror(errno));
    serverSocket.reset();
    return false;
  }

  if (listen(serverSocket.get(), 1) < 0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to listen: {}", strerror(errno));
    serverSocket.reset();
    return false;
  }

  TT_LOG_INFO("[TcpSocketTransport] Server initialized on port {}", port);
  return true;
}

bool TcpSocketTransport::initializeAsClient(const std::string& host,
                                            uint16_t port) {
  mode = Mode::CLIENT;
  this->host = host;
  this->port = port;

  TT_LOG_INFO("[TcpSocketTransport] Client initialized to connect to {}:{}",
              host, port);
  return true;
}

void TcpSocketTransport::start() {
  if (running) {
    return;
  }

  running = true;

  if (mode == Mode::SERVER) {
    connectionThread = std::jthread(
        [this](std::stop_token stopToken) { serverLoop(stopToken); });
  } else {
    connectionThread = std::jthread(
        [this](std::stop_token stopToken) { clientLoop(stopToken); });
  }
}

void TcpSocketTransport::stop() {
  if (!running) {
    return;
  }

  running = false;
  connected = false;

  {
    std::lock_guard<std::mutex> lock(socketMutex);
    int fd = peerSocket.load(std::memory_order_acquire);
    if (fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
    }
    peerSocket.store(-1, std::memory_order_release);
  }

  if (serverSocket) {
    ::shutdown(serverSocket.get(), SHUT_RDWR);
  }

  if (connectionThread.joinable()) {
    connectionThread.request_stop();
    connectionThread.join();
  }

  {
    std::lock_guard<std::mutex> lock(socketMutex);
    clientSocket.reset();
    serverSocket.reset();
  }

  TT_LOG_INFO("[TcpSocketTransport] Stopped");
}

void TcpSocketTransport::serverLoop(std::stop_token stopToken) {
  while (running && !stopToken.stop_requested()) {
    TT_LOG_INFO("[TcpSocketTransport] Waiting for client connection...");

    struct sockaddr_in clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    tt::utils::ScopedFd accepted =
        acceptClient(serverSocket.get(), &clientAddr, &clientLen);
    if (!accepted) {
      if (running && !stopToken.stop_requested()) {
        TT_LOG_ERROR("[TcpSocketTransport] Accept failed: {}", strerror(errno));
      }
      break;
    }

    configureSocket(accepted.get());

    peerSocket.store(accepted.get(), std::memory_order_release);
    connected = true;

    TT_LOG_INFO("[TcpSocketTransport] Client connected from {}:{}",
                inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));
    notifyConnectionEstablished();

    while (running && connected && !stopToken.stop_requested()) {
      std::this_thread::sleep_for(CONNECTION_POLL_INTERVAL);
    }

    {
      std::lock_guard<std::mutex> lock(socketMutex);
      peerSocket.store(-1, std::memory_order_release);
      accepted.reset();
    }
    markDisconnected();
    notifyConnectionLost();

    TT_LOG_INFO("[TcpSocketTransport] Client disconnected");
  }
}

void TcpSocketTransport::clientLoop(std::stop_token stopToken) {
  auto delay = reconnectInitialDelay;
  auto backoff = [&]() {
    std::this_thread::sleep_for(delay);
    delay = std::min(delay * 2, reconnectMaxDelay);
  };

  while (running && !stopToken.stop_requested()) {
    clientSocket = createTcpSocket();
    if (!clientSocket) {
      TT_LOG_ERROR("[TcpSocketTransport] Failed to create client socket: {}",
                   strerror(errno));
      backoff();
      continue;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr) <= 0) {
      TT_LOG_ERROR("[TcpSocketTransport] Invalid address: {}", host);
      clientSocket.reset();
      backoff();
      continue;
    }

    TT_LOG_INFO(
        "[TcpSocketTransport] Attempting to connect to {}:{} (backoff {}ms)",
        host, port, delay.count());

    if (connect(clientSocket.get(), (struct sockaddr*)&serverAddr,
                sizeof(serverAddr)) < 0) {
      TT_LOG_ERROR("[TcpSocketTransport] Connection failed: {}",
                   strerror(errno));
      clientSocket.reset();
      backoff();
      continue;
    }

    configureSocket(clientSocket.get());

    peerSocket.store(clientSocket.get(), std::memory_order_release);
    connected = true;
    delay = reconnectInitialDelay;  // reset on success

    TT_LOG_INFO("[TcpSocketTransport] Connected to server");
    notifyConnectionEstablished();

    while (running && connected && !stopToken.stop_requested()) {
      std::this_thread::sleep_for(CONNECTION_POLL_INTERVAL);
    }

    {
      std::lock_guard<std::mutex> lock(socketMutex);
      peerSocket.store(-1, std::memory_order_release);
      clientSocket.reset();
    }
    markDisconnected();
    notifyConnectionLost();

    TT_LOG_INFO("[TcpSocketTransport] Disconnected from server");
  }
}

bool TcpSocketTransport::sendRawData(std::span<const uint8_t> data) {
  std::lock_guard<std::mutex> lock(socketMutex);
  if (!connected) return false;

  int fd = peerSocket.load(std::memory_order_acquire);
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
  // Flatten the tri-state to the legacy empty-buffer contract: both NO_DATA and
  // CLOSED come back empty. Callers that need the distinction (the KV-migration
  // control channel) use tryReceiveMessage() instead.
  return std::move(tryReceiveMessage().data);
}

ReceiveResult TcpSocketTransport::tryReceiveMessage() {
  std::lock_guard<std::mutex> lock(socketMutex);
  if (!connected) return {ReceiveStatus::CLOSED, {}};

  int fd = peerSocket.load(std::memory_order_acquire);
  if (fd < 0) return {ReceiveStatus::CLOSED, {}};

  uint32_t netSize = 0;
  auto headerStatus =
      receiveExact(fd, reinterpret_cast<uint8_t*>(&netSize), sizeof(netSize),
                   MAX_HEADER_RETRIES, /*returnIfNoInitialData=*/true);
  if (headerStatus == ReadResult::NO_DATA) {
    return {ReceiveStatus::NO_DATA, {}};
  }
  if (headerStatus == ReadResult::DISCONNECTED) {
    markDisconnected();
    return {ReceiveStatus::CLOSED, {}};
  }

  uint32_t size = ntohl(netSize);
  if (size == 0 || size > MAX_MESSAGE_SIZE_BYTES) {
    markDisconnected();
    return {ReceiveStatus::CLOSED, {}};
  }

  std::vector<uint8_t> data(size);
  auto payloadStatus =
      receiveExact(fd, data.data(), data.size(), MAX_PAYLOAD_RETRIES,
                   /*returnIfNoInitialData=*/false);
  if (payloadStatus != ReadResult::COMPLETE) {
    markDisconnected();
    return {ReceiveStatus::CLOSED, {}};
  }

  return {ReceiveStatus::DATA, std::move(data)};
}

TcpSocketTransport::ReadResult TcpSocketTransport::receiveExact(
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
      return ReadResult::DISCONNECTED;
    }

    if (receivedTotal == 0 && returnIfNoInitialData) {
      return ReadResult::NO_DATA;
    }

    if (++retries > maxRetries) {
      return ReadResult::DISCONNECTED;
    }
    std::this_thread::sleep_for(RETRY_SLEEP);
  }

  return ReadResult::COMPLETE;
}

bool TcpSocketTransport::isConnected() const { return isConnectedState(); }

std::string TcpSocketTransport::getStatus() const {
  return getStatusString(isConnectedState());
}

void TcpSocketTransport::setConnectionLostCallback(
    std::function<void()> callback) {
  setConnectionLostCallbackCommon(std::move(callback));
}

void TcpSocketTransport::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  setConnectionEstablishedCallbackCommon(std::move(callback));
}

void TcpSocketTransport::setReconnectBackoff(
    std::chrono::milliseconds initialDelay,
    std::chrono::milliseconds maxDelay) {
  setReconnectBackoffCommon(initialDelay, maxDelay);
}

}  // namespace tt::sockets
