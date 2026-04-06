// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "sockets/socket_manager.hpp"

#include <errno.h>
#include <fcntl.h>
#include <netinet/tcp.h>

#include <cstring>

#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
void setSocketKeepAlive(int socketFd) {
  int enable = 1;
  if (setsockopt(socketFd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) <
      0) {
    TT_LOG_WARN("[SocketManager] Failed to set SO_KEEPALIVE: {}",
                strerror(errno));
  }

  int idle = 10;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle)) <
      0) {
    TT_LOG_WARN("[SocketManager] Failed to set TCP_KEEPIDLE: {}",
                strerror(errno));
  }

  int interval = 5;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPINTVL, &interval,
                 sizeof(interval)) < 0) {
    TT_LOG_WARN("[SocketManager] Failed to set TCP_KEEPINTVL: {}",
                strerror(errno));
  }

  int maxProbes = 3;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPCNT, &maxProbes,
                 sizeof(maxProbes)) < 0) {
    TT_LOG_WARN("[SocketManager] Failed to set TCP_KEEPCNT: {}",
                strerror(errno));
  }
}
}  // namespace

SocketManager& SocketManager::getInstance() {
  static SocketManager instance;
  return instance;
}

SocketManager::~SocketManager() { stop(); }

bool SocketManager::initializeAsServer(uint16_t port) {
  mode_ = Mode::SERVER;
  port_ = port;

  server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket_ < 0) {
    TT_LOG_ERROR("[SocketManager] Failed to create server socket: {}",
                 strerror(errno));
    return false;
  }

  // Set socket options
  int opt = 1;
  if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) <
      0) {
    TT_LOG_ERROR("[SocketManager] Failed to set SO_REUSEADDR: {}",
                 strerror(errno));
    close(server_socket_);
    return false;
  }

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port_);

  if (bind(server_socket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
    TT_LOG_ERROR("[SocketManager] Failed to bind to port {}: {}", port_,
                 strerror(errno));
    close(server_socket_);
    return false;
  }

  if (listen(server_socket_, 1) < 0) {
    TT_LOG_ERROR("[SocketManager] Failed to listen: {}", strerror(errno));
    close(server_socket_);
    return false;
  }

  TT_LOG_INFO("[SocketManager] Server initialized on port {}", port_);
  return true;
}

bool SocketManager::initializeAsClient(const std::string& host, uint16_t port) {
  mode_ = Mode::CLIENT;
  host_ = host;
  port_ = port;

  TT_LOG_INFO("[SocketManager] Client initialized to connect to {}:{}", host_,
              port_);
  return true;
}

void SocketManager::start() {
  if (running_) {
    return;
  }

  running_ = true;

  if (mode_ == Mode::SERVER) {
    server_thread_ = std::thread(&SocketManager::serverLoop, this);
  } else {
    server_thread_ = std::thread(&SocketManager::clientLoop, this);
  }

  message_thread_ = std::thread(&SocketManager::messageLoop, this);
}

void SocketManager::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  connected_ = false;

  if (peer_socket_ >= 0) {
    close(peer_socket_);
    peer_socket_ = -1;
  }

  if (client_socket_ >= 0) {
    close(client_socket_);
    client_socket_ = -1;
  }

  if (server_socket_ >= 0) {
    close(server_socket_);
    server_socket_ = -1;
  }

  if (server_thread_.joinable()) {
    server_thread_.join();
  }

  if (message_thread_.joinable()) {
    message_thread_.join();
  }

  TT_LOG_INFO("[SocketManager] Stopped");
}

void SocketManager::serverLoop() {
  while (running_) {
    TT_LOG_INFO("[SocketManager] Waiting for client connection...");

    struct sockaddr_in clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    int newSocket =
        accept(server_socket_, (struct sockaddr*)&clientAddr, &clientLen);
    if (newSocket < 0) {
      if (running_) {
        TT_LOG_ERROR("[SocketManager] Accept failed: {}", strerror(errno));
      }
      break;
    }

    int flags = fcntl(newSocket, F_GETFL, 0);
    if (flags < 0 || fcntl(newSocket, F_SETFL, flags | O_NONBLOCK) < 0) {
      TT_LOG_WARN("[SocketManager] Failed to set non-blocking: {}",
                  strerror(errno));
    }
    setSocketKeepAlive(newSocket);

    peer_socket_ = newSocket;
    connected_ = true;

    TT_LOG_INFO("[SocketManager] Client connected from {}:{}",
                inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));

    // Wait until disconnected
    while (running_ && connected_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (peer_socket_ >= 0) {
      close(peer_socket_);
      peer_socket_ = -1;
    }
    bool wasConnected = connected_.exchange(false);
    if (wasConnected && connection_lost_callback_) {
      connection_lost_callback_();
    }

    TT_LOG_INFO("[SocketManager] Client disconnected");
  }
}

void SocketManager::clientLoop() {
  while (running_) {
    client_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket_ < 0) {
      TT_LOG_ERROR("[SocketManager] Failed to create client socket: {}",
                   strerror(errno));
      std::this_thread::sleep_for(std::chrono::seconds(5));
      continue;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &serverAddr.sin_addr) <= 0) {
      TT_LOG_ERROR("[SocketManager] Invalid address: {}", host_);
      close(client_socket_);
      std::this_thread::sleep_for(std::chrono::seconds(5));
      continue;
    }

    TT_LOG_INFO("[SocketManager] Attempting to connect to {}:{}", host_, port_);

    if (connect(client_socket_, (struct sockaddr*)&serverAddr,
                sizeof(serverAddr)) < 0) {
      TT_LOG_ERROR("[SocketManager] Connection failed: {}", strerror(errno));
      close(client_socket_);
      std::this_thread::sleep_for(std::chrono::seconds(5));
      continue;
    }

    int flags = fcntl(client_socket_, F_GETFL, 0);
    if (flags < 0 || fcntl(client_socket_, F_SETFL, flags | O_NONBLOCK) < 0) {
      TT_LOG_WARN("[SocketManager] Failed to set non-blocking: {}",
                  strerror(errno));
    }
    setSocketKeepAlive(client_socket_);

    peer_socket_ = client_socket_;
    connected_ = true;

    TT_LOG_INFO("[SocketManager] Connected to server");

    // Wait until disconnected
    while (running_ && connected_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (client_socket_ >= 0) {
      close(client_socket_);
      client_socket_ = -1;
    }
    peer_socket_ = -1;
    bool wasConnected = connected_.exchange(false);
    if (wasConnected && connection_lost_callback_) {
      connection_lost_callback_();
    }

    TT_LOG_INFO("[SocketManager] Disconnected from server");

    if (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  }
}

void SocketManager::messageLoop() {
  while (running_) {
    if (!connected_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    try {
      auto data = receiveRawData();
      if (!data.empty()) {
        handleIncomingMessage(data);
      }
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[SocketManager] Message loop error: {}", e.what());
      connected_ = false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

bool SocketManager::sendRawData(const std::vector<uint8_t>& data) {
  if (!connected_ || peer_socket_ < 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(send_mutex_);

  // Send data size first
  uint32_t size = static_cast<uint32_t>(data.size());
  uint32_t netSize = htonl(size);

  ssize_t sent = send(peer_socket_, &netSize, sizeof(netSize), MSG_NOSIGNAL);
  if (sent != sizeof(netSize)) {
    connected_ = false;
    return false;
  }

  // Send actual data
  size_t totalSent = 0;
  while (totalSent < data.size()) {
    sent = send(peer_socket_, data.data() + totalSent, data.size() - totalSent,
                MSG_NOSIGNAL);
    if (sent <= 0) {
      connected_ = false;
      return false;
    }
    totalSent += sent;
  }

  return true;
}

std::vector<uint8_t> SocketManager::receiveRawData() {
  if (!connected_ || peer_socket_ < 0) {
    return {};
  }

  // Read data size first
  uint32_t netSize;
  ssize_t received =
      recv(peer_socket_, &netSize, sizeof(netSize), MSG_DONTWAIT);
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

  // Read actual data
  std::vector<uint8_t> data(size);
  size_t totalReceived = 0;
  int retryCount = 0;
  const int maxRetries = 1000;  // 1 second timeout (1000 * 1ms)

  while (totalReceived < size) {
    received = recv(peer_socket_, data.data() + totalReceived,
                    size - totalReceived, 0);
    if (received > 0) {
      totalReceived += received;
      retryCount = 0;  // Reset retry count on successful receive
    } else if (received == 0) {
      // Connection closed by peer
      connected_ = false;
      return {};
    } else {
      // received < 0: check if it's a temporary error
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // Non-blocking socket has no data yet, wait briefly and retry
        if (++retryCount > maxRetries) {
          TT_LOG_ERROR("[SocketManager] Timeout waiting for message data");
          connected_ = false;
          return {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      // Real error
      connected_ = false;
      return {};
    }
  }

  return data;
}

void SocketManager::handleIncomingMessage(const std::vector<uint8_t>& data) {
  try {
    // First, extract the message type
    std::string serialized(data.begin(), data.end());
    std::istringstream iss(serialized);

    cereal::BinaryInputArchive archive(iss);
    std::string messageType;
    archive(messageType);  // Use operator() instead of loadBinaryValue

    // Find handler for this message type
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    auto it = handlers_.find(messageType);
    if (it != handlers_.end()) {
      it->second(data);
    } else {
      TT_LOG_DEBUG("[SocketManager] No handler for message type: {}",
                   messageType);
    }
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SocketManager] Message handling error: {}", e.what());
  }
}

bool SocketManager::isConnected() const { return connected_; }

std::string SocketManager::getStatus() const {
  if (!running_) {
    return "stopped";
  }

  if (connected_) {
    return mode_ == Mode::SERVER ? "server:connected" : "client:connected";
  }

  return mode_ == Mode::SERVER ? "server:waiting" : "client:connecting";
}

void SocketManager::setConnectionLostCallback(std::function<void()> callback) {
  connection_lost_callback_ = std::move(callback);
}

}  // namespace tt::sockets
