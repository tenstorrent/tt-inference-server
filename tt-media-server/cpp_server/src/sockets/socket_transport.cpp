// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/socket_transport.hpp"

#include <errno.h>
#include <fcntl.h>
#include <netinet/tcp.h>

#include <algorithm>
#include <cstring>

#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
void setSocketKeepAlive(int socketFd) {
  int enable = 1;
  if (setsockopt(socketFd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) <
      0) {
    TT_LOG_WARN("[SocketTransport] Failed to set SO_KEEPALIVE: {}",
                strerror(errno));
  }

  int idle = 10;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle)) <
      0) {
    TT_LOG_WARN("[SocketTransport] Failed to set TCP_KEEPIDLE: {}",
                strerror(errno));
  }

  int interval = 5;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPINTVL, &interval,
                 sizeof(interval)) < 0) {
    TT_LOG_WARN("[SocketTransport] Failed to set TCP_KEEPINTVL: {}",
                strerror(errno));
  }

  int maxProbes = 3;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPCNT, &maxProbes,
                 sizeof(maxProbes)) < 0) {
    TT_LOG_WARN("[SocketTransport] Failed to set TCP_KEEPCNT: {}",
                strerror(errno));
  }
}

void setNonBlocking(int socketFd) {
  int flags = fcntl(socketFd, F_GETFL, 0);
  if (flags < 0 || fcntl(socketFd, F_SETFL, flags | O_NONBLOCK) < 0) {
    TT_LOG_WARN("[SocketTransport] Failed to set non-blocking: {}",
                strerror(errno));
  }
}

void configureSocket(int socketFd) {
  setNonBlocking(socketFd);
  setSocketKeepAlive(socketFd);
}
}  // namespace

SocketTransport::~SocketTransport() { stop(); }

bool SocketTransport::initializeAsServer(uint16_t port) {
  mode_ = Mode::SERVER;
  port_ = port;

  serverSocket_.reset(socket(AF_INET, SOCK_STREAM, 0));
  if (!serverSocket_) {
    TT_LOG_ERROR("[SocketTransport] Failed to create server socket: {}",
                 strerror(errno));
    return false;
  }

  int opt = 1;
  if (setsockopt(serverSocket_.get(), SOL_SOCKET, SO_REUSEADDR, &opt,
                 sizeof(opt)) < 0) {
    TT_LOG_ERROR("[SocketTransport] Failed to set SO_REUSEADDR: {}",
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
    TT_LOG_ERROR("[SocketTransport] Failed to bind to port {}: {}", port_,
                 strerror(errno));
    serverSocket_.reset();
    return false;
  }

  if (listen(serverSocket_.get(), 1) < 0) {
    TT_LOG_ERROR("[SocketTransport] Failed to listen: {}", strerror(errno));
    serverSocket_.reset();
    return false;
  }

  TT_LOG_INFO("[SocketTransport] Server initialized on port {}", port_);
  return true;
}

bool SocketTransport::initializeAsClient(const std::string& host,
                                         uint16_t port) {
  mode_ = Mode::CLIENT;
  host_ = host;
  port_ = port;

  TT_LOG_INFO("[SocketTransport] Client initialized to connect to {}:{}", host_,
              port_);
  return true;
}

void SocketTransport::start() {
  if (running_) {
    return;
  }

  running_ = true;

  if (mode_ == Mode::SERVER) {
    connectionThread_ = std::thread(&SocketTransport::serverLoop, this);
  } else {
    connectionThread_ = std::thread(&SocketTransport::clientLoop, this);
  }
}

void SocketTransport::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  connected_ = false;

  peerSocket_ = -1;

  // shutdown() unblocks any in-flight accept/connect/recv so the join() below
  // doesn't deadlock when stop() races with a thread parked in a syscall.
  if (serverSocket_) {
    ::shutdown(serverSocket_.get(), SHUT_RDWR);
  }
  if (clientSocket_) {
    ::shutdown(clientSocket_.get(), SHUT_RDWR);
  }

  clientSocket_.reset();
  serverSocket_.reset();

  if (connectionThread_.joinable()) {
    connectionThread_.join();
  }

  TT_LOG_INFO("[SocketTransport] Stopped");
}

void SocketTransport::serverLoop() {
  while (running_) {
    TT_LOG_INFO("[SocketTransport] Waiting for client connection...");

    struct sockaddr_in clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    tt::utils::ScopedFd accepted(
        accept(serverSocket_.get(), (struct sockaddr*)&clientAddr, &clientLen));
    if (!accepted) {
      if (running_) {
        TT_LOG_ERROR("[SocketTransport] Accept failed: {}", strerror(errno));
      }
      break;
    }

    configureSocket(accepted.get());

    peerSocket_ = accepted.get();
    connected_ = true;

    TT_LOG_INFO("[SocketTransport] Client connected from {}:{}",
                inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));

    if (connectionEstablishedCallback_) {
      connectionEstablishedCallback_();
    }

    while (running_ && connected_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    accepted.reset();
    peerSocket_ = -1;
    // We were connected when we entered the spin, so the transition out always
    // represents a connection loss. Not using exchange() here: send/recv error
    // paths flip connected_=false too and would race the callback away.
    connected_ = false;
    if (connectionLostCallback_) {
      connectionLostCallback_();
    }

    TT_LOG_INFO("[SocketTransport] Client disconnected");
  }
}

void SocketTransport::clientLoop() {
  uint32_t delayMs = reconnectInitialDelayMs_;
  auto backoff = [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    delayMs = std::min(delayMs * 2, reconnectMaxDelayMs_);
  };

  while (running_) {
    clientSocket_.reset(socket(AF_INET, SOCK_STREAM, 0));
    if (!clientSocket_) {
      TT_LOG_ERROR("[SocketTransport] Failed to create client socket: {}",
                   strerror(errno));
      backoff();
      continue;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &serverAddr.sin_addr) <= 0) {
      TT_LOG_ERROR("[SocketTransport] Invalid address: {}", host_);
      clientSocket_.reset();
      backoff();
      continue;
    }

    TT_LOG_INFO("[SocketTransport] Attempting to connect to {}:{} (backoff {}ms)",
                host_, port_, delayMs);

    if (connect(clientSocket_.get(), (struct sockaddr*)&serverAddr,
                sizeof(serverAddr)) < 0) {
      TT_LOG_ERROR("[SocketTransport] Connection failed: {}", strerror(errno));
      clientSocket_.reset();
      backoff();
      continue;
    }

    configureSocket(clientSocket_.get());

    peerSocket_ = clientSocket_.get();
    connected_ = true;
    delayMs = reconnectInitialDelayMs_;  // reset on success

    TT_LOG_INFO("[SocketTransport] Connected to server");

    if (connectionEstablishedCallback_) {
      connectionEstablishedCallback_();
    }

    while (running_ && connected_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    peerSocket_ = -1;
    clientSocket_.reset();
    // See serverLoop comment: always-fire on exit-from-connected.
    connected_ = false;
    if (connectionLostCallback_) {
      connectionLostCallback_();
    }

    TT_LOG_INFO("[SocketTransport] Disconnected from server");
  }
}

bool SocketTransport::sendRawData(const std::vector<uint8_t>& data) {
  if (!connected_ || peerSocket_ < 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(sendMutex_);

  uint32_t size = static_cast<uint32_t>(data.size());
  uint32_t netSize = htonl(size);

  ssize_t sent = send(peerSocket_, &netSize, sizeof(netSize), MSG_NOSIGNAL);
  if (sent != sizeof(netSize)) {
    connected_ = false;
    return false;
  }

  size_t totalSent = 0;
  while (totalSent < data.size()) {
    sent = send(peerSocket_, data.data() + totalSent, data.size() - totalSent,
                MSG_NOSIGNAL);
    if (sent <= 0) {
      connected_ = false;
      return false;
    }
    totalSent += sent;
  }

  return true;
}

std::vector<uint8_t> SocketTransport::receiveRawData() {
  if (!connected_ || peerSocket_ < 0) {
    return {};
  }

  uint32_t netSize;
  ssize_t received = recv(peerSocket_, &netSize, sizeof(netSize), MSG_DONTWAIT);
  if (received <= 0) {
    if (received == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
      connected_ = false;
    }
    return {};
  }

  if (received != sizeof(netSize)) {
    connected_ = false;
    return {};
  }

  uint32_t size = ntohl(netSize);
  if (size == 0 || size > 1024 * 1024) {  // Max 1MB per message
    connected_ = false;
    return {};
  }

  std::vector<uint8_t> data(size);
  size_t totalReceived = 0;
  int retryCount = 0;
  const int maxRetries = 1000;  // 1 second timeout (1000 * 1ms)

  while (totalReceived < size) {
    received =
        recv(peerSocket_, data.data() + totalReceived, size - totalReceived, 0);
    if (received > 0) {
      totalReceived += received;
      retryCount = 0;
    } else if (received == 0) {
      connected_ = false;
      return {};
    } else {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (++retryCount > maxRetries) {
          TT_LOG_ERROR("[SocketTransport] Timeout waiting for message data");
          connected_ = false;
          return {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      connected_ = false;
      return {};
    }
  }

  return data;
}

bool SocketTransport::isConnected() const { return connected_; }

std::string SocketTransport::getStatus() const {
  if (!running_) {
    return "stopped";
  }

  if (connected_) {
    return mode_ == Mode::SERVER ? "server:connected" : "client:connected";
  }

  return mode_ == Mode::SERVER ? "server:waiting" : "client:connecting";
}

void SocketTransport::setConnectionLostCallback(
    std::function<void()> callback) {
  connectionLostCallback_ = std::move(callback);
}

void SocketTransport::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  connectionEstablishedCallback_ = std::move(callback);
}

void SocketTransport::setReconnectBackoff(uint32_t initialDelayMs,
                                          uint32_t maxDelayMs) {
  reconnectInitialDelayMs_ = initialDelayMs;
  reconnectMaxDelayMs_ = maxDelayMs;
}

}  // namespace tt::sockets
