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
  setsockopt(socketFd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable));

  int idle = 10;
  setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));

  int interval = 5;
  setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));

  int maxProbes = 3;
  setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPCNT, &maxProbes, sizeof(maxProbes));
}
}  // namespace

SocketManager& SocketManager::getInstance() {
  static SocketManager instance;
  return instance;
}

SocketManager::~SocketManager() { stop(); }

bool SocketManager::initializeAsServer(uint16_t port) {
  mode = Mode::SERVER;
  this->port = port;

  server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket < 0) {
    TT_LOG_ERROR("[SocketManager] Failed to create server socket: {}",
                 strerror(errno));
    return false;
  }

  // Set socket options
  int opt = 1;
  if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) <
      0) {
    TT_LOG_ERROR("[SocketManager] Failed to set SO_REUSEADDR: {}",
                 strerror(errno));
    close(server_socket);
    return false;
  }

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  if (bind(server_socket, (struct sockaddr*)&address, sizeof(address)) < 0) {
    TT_LOG_ERROR("[SocketManager] Failed to bind to port {}: {}", port,
                 strerror(errno));
    close(server_socket);
    return false;
  }

  if (listen(server_socket, 1) < 0) {
    TT_LOG_ERROR("[SocketManager] Failed to listen: {}", strerror(errno));
    close(server_socket);
    return false;
  }

  TT_LOG_INFO("[SocketManager] Server initialized on port {}", port);
  return true;
}

bool SocketManager::initializeAsClient(const std::string& host, uint16_t port) {
  mode = Mode::CLIENT;
  this->host = host;
  this->port = port;

  TT_LOG_INFO("[SocketManager] Client initialized to connect to {}:{}", host,
              port);
  return true;
}

void SocketManager::start() {
  if (running) {
    return;
  }

  running = true;

  if (mode == Mode::SERVER) {
    server_thread = std::thread(&SocketManager::serverLoop, this);
  } else {
    server_thread = std::thread(&SocketManager::clientLoop, this);
  }

  message_thread = std::thread(&SocketManager::messageLoop, this);
}

void SocketManager::stop() {
  if (!running) {
    return;
  }

  running = false;
  connected = false;

  if (peer_socket >= 0) {
    close(peer_socket);
    peer_socket = -1;
  }

  if (client_socket >= 0) {
    close(client_socket);
    client_socket = -1;
  }

  if (server_socket >= 0) {
    close(server_socket);
    server_socket = -1;
  }

  if (server_thread.joinable()) {
    server_thread.join();
  }

  if (message_thread.joinable()) {
    message_thread.join();
  }

  TT_LOG_INFO("[SocketManager] Stopped");
}

void SocketManager::serverLoop() {
  while (running) {
    TT_LOG_INFO("[SocketManager] Waiting for client connection...");

    struct sockaddr_in clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    int newSocket =
        accept(server_socket, (struct sockaddr*)&clientAddr, &clientLen);
    if (newSocket < 0) {
      if (running) {
        TT_LOG_ERROR("[SocketManager] Accept failed: {}", strerror(errno));
      }
      break;
    }

    // Set non-blocking mode and keep-alive
    int flags = fcntl(newSocket, F_GETFL, 0);
    fcntl(newSocket, F_SETFL, flags | O_NONBLOCK);
    setSocketKeepAlive(newSocket);

    peer_socket = newSocket;
    connected = true;

    TT_LOG_INFO("[SocketManager] Client connected from {}:{}",
                inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));

    // Wait until disconnected
    while (running && connected) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (peer_socket >= 0) {
      close(peer_socket);
      peer_socket = -1;
    }
    bool wasConnected = connected.exchange(false);
    if (wasConnected && connection_lost_callback) {
      connection_lost_callback();
    }

    TT_LOG_INFO("[SocketManager] Client disconnected");
  }
}

void SocketManager::clientLoop() {
  while (running) {
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
      TT_LOG_ERROR("[SocketManager] Failed to create client socket: {}",
                   strerror(errno));
      std::this_thread::sleep_for(std::chrono::seconds(5));
      continue;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr) <= 0) {
      TT_LOG_ERROR("[SocketManager] Invalid address: {}", host);
      close(client_socket);
      std::this_thread::sleep_for(std::chrono::seconds(5));
      continue;
    }

    TT_LOG_INFO("[SocketManager] Attempting to connect to {}:{}", host, port);

    if (connect(client_socket, (struct sockaddr*)&serverAddr,
                sizeof(serverAddr)) < 0) {
      TT_LOG_ERROR("[SocketManager] Connection failed: {}", strerror(errno));
      close(client_socket);
      std::this_thread::sleep_for(std::chrono::seconds(5));
      continue;
    }

    // Set non-blocking mode and keep-alive
    int flags = fcntl(client_socket, F_GETFL, 0);
    fcntl(client_socket, F_SETFL, flags | O_NONBLOCK);
    setSocketKeepAlive(client_socket);

    peer_socket = client_socket;
    connected = true;

    TT_LOG_INFO("[SocketManager] Connected to server");

    // Wait until disconnected
    while (running && connected) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (client_socket >= 0) {
      close(client_socket);
      client_socket = -1;
    }
    peer_socket = -1;
    bool wasConnected = connected.exchange(false);
    if (wasConnected && connection_lost_callback) {
      connection_lost_callback();
    }

    TT_LOG_INFO("[SocketManager] Disconnected from server");

    if (running) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  }
}

void SocketManager::messageLoop() {
  while (running) {
    if (!connected) {
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
      connected = false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

bool SocketManager::sendRawData(const std::vector<uint8_t>& data) {
  if (!connected || peer_socket < 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(send_mutex);

  // Send data size first
  uint32_t size = static_cast<uint32_t>(data.size());
  uint32_t netSize = htonl(size);

  ssize_t sent = send(peer_socket, &netSize, sizeof(netSize), MSG_NOSIGNAL);
  if (sent != sizeof(netSize)) {
    connected = false;
    return false;
  }

  // Send actual data
  size_t totalSent = 0;
  while (totalSent < data.size()) {
    sent = send(peer_socket, data.data() + totalSent, data.size() - totalSent,
                MSG_NOSIGNAL);
    if (sent <= 0) {
      connected = false;
      return false;
    }
    totalSent += sent;
  }

  return true;
}

std::vector<uint8_t> SocketManager::receiveRawData() {
  if (!connected || peer_socket < 0) {
    return {};
  }

  // Read data size first
  uint32_t netSize;
  ssize_t received =
      recv(peer_socket, &netSize, sizeof(netSize), MSG_DONTWAIT);
  if (received <= 0) {
    if (received == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
      connected = false;
    }
    return {};
  }

  if (received != sizeof(netSize)) {
    connected = false;
    return {};
  }

  uint32_t size = ntohl(netSize);
  if (size == 0 || size > 1024 * 1024) {  // Max 1MB per message
    connected = false;
    return {};
  }

  // Read actual data
  std::vector<uint8_t> data(size);
  size_t totalReceived = 0;
  int retryCount = 0;
  const int MAX_RETRIES = 1000;  // 1 second timeout (1000 * 1ms)

  while (totalReceived < size) {
    received = recv(peer_socket, data.data() + totalReceived,
                    size - totalReceived, 0);
    if (received > 0) {
      totalReceived += received;
      retryCount = 0;  // Reset retry count on successful receive
    } else if (received == 0) {
      // Connection closed by peer
      connected = false;
      return {};
    } else {
      // received < 0: check if it's a temporary error
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // Non-blocking socket has no data yet, wait briefly and retry
        if (++retryCount > MAX_RETRIES) {
          TT_LOG_ERROR("[SocketManager] Timeout waiting for message data");
          connected = false;
          return {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      // Real error
      connected = false;
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
    std::lock_guard<std::mutex> lock(handlers_mutex);
    auto it = handlers.find(messageType);
    if (it != handlers.end()) {
      it->second(data);
    } else {
      TT_LOG_DEBUG("[SocketManager] No handler for message type: {}",
                   messageType);
    }
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SocketManager] Message handling error: {}", e.what());
  }
}

bool SocketManager::isConnected() const { return connected; }

std::string SocketManager::getStatus() const {
  if (!running) {
    return "stopped";
  }

  if (connected) {
    return mode == Mode::SERVER ? "server:connected" : "client:connected";
  }

  return mode == Mode::SERVER ? "server:waiting" : "client:connecting";
}

void SocketManager::setConnectionLostCallback(std::function<void()> callback) {
  connection_lost_callback = std::move(callback);
}

}  // namespace tt::sockets
