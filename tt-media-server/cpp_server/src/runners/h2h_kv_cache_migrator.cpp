// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/h2h_kv_cache_migrator.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>

#include <cstring>

#include "utils/logger.hpp"

namespace llm_engine {

namespace {
constexpr int RECONNECT_INTERVAL_S = 5;
constexpr int RECV_TIMEOUT_S = 1;
}  // namespace

H2HKVCacheMigrator::H2HKVCacheMigrator(Mode mode, const std::string& host,
                                       uint16_t port)
    : mode_(mode), host_(host), port_(port) {}

H2HKVCacheMigrator::~H2HKVCacheMigrator() { stop(); }

void H2HKVCacheMigrator::setReceiveCallback(ReceiveCallback cb) {
  receive_callback_ = std::move(cb);
}

void H2HKVCacheMigrator::start() {
  if (running_.exchange(true)) return;

  if (mode_ == Mode::SERVER) {
    connect_thread_ = std::thread(&H2HKVCacheMigrator::serverLoop, this);
  } else {
    connect_thread_ = std::thread(&H2HKVCacheMigrator::clientLoop, this);
  }

  TT_LOG_INFO("[H2HKVCacheMigrator] Started (mode={}, {}:{})",
              mode_ == Mode::SERVER ? "server" : "client", host_, port_);
}

void H2HKVCacheMigrator::stop() {
  if (!running_.exchange(false)) return;

  connected_ = false;

  if (peer_fd_ >= 0) {
    ::close(peer_fd_);
    peer_fd_ = -1;
  }
  if (server_fd_ >= 0) {
    ::close(server_fd_);
    server_fd_ = -1;
  }

  if (connect_thread_.joinable()) connect_thread_.join();
  if (receive_thread_.joinable()) receive_thread_.join();

  TT_LOG_INFO("[H2HKVCacheMigrator] Stopped");
}

// ---------------------------------------------------------------------------
// Connection management
// ---------------------------------------------------------------------------

void H2HKVCacheMigrator::serverLoop() {
  server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    TT_LOG_ERROR("[H2HKVCacheMigrator] socket(): {}", strerror(errno));
    return;
  }

  int opt = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port_);

  if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    TT_LOG_ERROR("[H2HKVCacheMigrator] bind({}): {}", port_, strerror(errno));
    ::close(server_fd_);
    server_fd_ = -1;
    return;
  }

  if (::listen(server_fd_, 1) < 0) {
    TT_LOG_ERROR("[H2HKVCacheMigrator] listen(): {}", strerror(errno));
    ::close(server_fd_);
    server_fd_ = -1;
    return;
  }

  while (running_) {
    TT_LOG_INFO("[H2HKVCacheMigrator] Waiting for prefill client on port {}...",
                port_);

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int fd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr),
                      &client_len);
    if (fd < 0) {
      if (running_) {
        TT_LOG_ERROR("[H2HKVCacheMigrator] accept(): {}", strerror(errno));
      }
      break;
    }

    peer_fd_ = fd;
    connected_ = true;
    TT_LOG_INFO("[H2HKVCacheMigrator] Client connected");

    receiveLoop();

    if (peer_fd_ >= 0) {
      ::close(peer_fd_);
      peer_fd_ = -1;
    }
    connected_ = false;
    TT_LOG_INFO("[H2HKVCacheMigrator] Client disconnected");
  }
}

void H2HKVCacheMigrator::clientLoop() {
  while (running_) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      TT_LOG_ERROR("[H2HKVCacheMigrator] socket(): {}", strerror(errno));
      std::this_thread::sleep_for(std::chrono::seconds(RECONNECT_INTERVAL_S));
      continue;
    }

    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string portStr = std::to_string(port_);
    struct addrinfo* addrResult = nullptr;
    int rc = getaddrinfo(host_.c_str(), portStr.c_str(), &hints, &addrResult);
    if (rc != 0 || addrResult == nullptr) {
      TT_LOG_ERROR("[H2HKVCacheMigrator] Cannot resolve '{}': {}", host_,
                   gai_strerror(rc));
      ::close(fd);
      std::this_thread::sleep_for(std::chrono::seconds(RECONNECT_INTERVAL_S));
      continue;
    }

    TT_LOG_INFO("[H2HKVCacheMigrator] Connecting to {}:{}...", host_, port_);

    int connectResult =
        ::connect(fd, addrResult->ai_addr, addrResult->ai_addrlen);
    freeaddrinfo(addrResult);

    if (connectResult < 0) {
      TT_LOG_ERROR("[H2HKVCacheMigrator] connect(): {}", strerror(errno));
      ::close(fd);
      std::this_thread::sleep_for(std::chrono::seconds(RECONNECT_INTERVAL_S));
      continue;
    }

    peer_fd_ = fd;
    connected_ = true;
    TT_LOG_INFO("[H2HKVCacheMigrator] Connected to decode server");

    receiveLoop();

    if (peer_fd_ >= 0) {
      ::close(peer_fd_);
      peer_fd_ = -1;
    }
    connected_ = false;
    TT_LOG_INFO("[H2HKVCacheMigrator] Disconnected from decode server");

    if (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(RECONNECT_INTERVAL_S));
    }
  }
}

void H2HKVCacheMigrator::receiveLoop() {
  // Set a receive timeout so we periodically check running_
  timeval tv{RECV_TIMEOUT_S, 0};
  setsockopt(peer_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  while (running_ && connected_) {
    uint32_t net_len = 0;
    if (!recvAll(peer_fd_, &net_len, sizeof(net_len))) {
      break;
    }

    uint32_t total_len = ntohl(net_len);
    if (total_len == 0) continue;

    std::vector<uint8_t> buf(total_len);
    if (!recvAll(peer_fd_, buf.data(), total_len)) {
      break;
    }

    try {
      auto data = deserialize(buf);
      TT_LOG_INFO(
          "[H2HKVCacheMigrator] Received KV cache for task {} ({} blocks, {} "
          "bytes payload)",
          data.task_id, data.block_ids.size(), data.payload.size());
      if (receive_callback_) {
        receive_callback_(std::move(data));
      }
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[H2HKVCacheMigrator] Deserialize error: {}", e.what());
    }
  }
}

// ---------------------------------------------------------------------------
// Send / receive helpers
// ---------------------------------------------------------------------------

bool H2HKVCacheMigrator::sendAll(int fd, const void* buf, size_t len) {
  auto* p = static_cast<const uint8_t*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, p + sent, len - sent, MSG_NOSIGNAL);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool H2HKVCacheMigrator::recvAll(int fd, void* buf, size_t len) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < len) {
    ssize_t n = ::recv(fd, p + got, len - got, 0);
    if (n > 0) {
      got += static_cast<size_t>(n);
    } else if (n == 0) {
      return false;
    } else {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (!running_) return false;
        continue;
      }
      return false;
    }
  }
  return true;
}

void H2HKVCacheMigrator::send(KVCacheMigrationData data) {
  if (!connected_ || peer_fd_ < 0) {
    TT_LOG_WARN("[H2HKVCacheMigrator] Cannot send: not connected");
    return;
  }

  auto buf = serialize(data);

  uint32_t net_len = htonl(static_cast<uint32_t>(buf.size()));
  std::lock_guard<std::mutex> lock(send_mutex_);

  if (!sendAll(peer_fd_, &net_len, sizeof(net_len)) ||
      !sendAll(peer_fd_, buf.data(), buf.size())) {
    TT_LOG_ERROR("[H2HKVCacheMigrator] Send failed for task {}", data.task_id);
    connected_ = false;
    return;
  }

  TT_LOG_INFO(
      "[H2HKVCacheMigrator] Sent KV cache for task {} ({} blocks, {} bytes)",
      data.task_id, data.block_ids.size(), buf.size());
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

std::vector<uint8_t> H2HKVCacheMigrator::serialize(
    const KVCacheMigrationData& data) {
  // Layout:
  //   [4B task_id_len][task_id]
  //   [4B num_blocks][block_ids (4B each)]
  //   [remaining: payload]
  uint32_t tid_len = static_cast<uint32_t>(data.task_id.size());
  uint32_t num_blocks = static_cast<uint32_t>(data.block_ids.size());
  size_t total = sizeof(tid_len) + tid_len + sizeof(num_blocks) +
                 num_blocks * sizeof(int32_t) + data.payload.size();

  std::vector<uint8_t> buf(total);
  uint8_t* p = buf.data();

  auto write32 = [&p](uint32_t v) {
    uint32_t nv = htonl(v);
    std::memcpy(p, &nv, 4);
    p += 4;
  };

  write32(tid_len);
  std::memcpy(p, data.task_id.data(), tid_len);
  p += tid_len;

  write32(num_blocks);
  for (int bid : data.block_ids) {
    write32(static_cast<uint32_t>(bid));
  }

  std::memcpy(p, data.payload.data(), data.payload.size());

  return buf;
}

KVCacheMigrationData H2HKVCacheMigrator::deserialize(
    const std::vector<uint8_t>& buf) {
  const uint8_t* p = buf.data();
  const uint8_t* end = p + buf.size();

  auto read32 = [&p, end]() -> uint32_t {
    if (p + 4 > end) throw std::runtime_error("truncated message");
    uint32_t nv;
    std::memcpy(&nv, p, 4);
    p += 4;
    return ntohl(nv);
  };

  KVCacheMigrationData data;

  uint32_t tid_len = read32();
  if (p + tid_len > end) throw std::runtime_error("truncated task_id");
  data.task_id.assign(reinterpret_cast<const char*>(p), tid_len);
  p += tid_len;

  uint32_t num_blocks = read32();
  data.block_ids.resize(num_blocks);
  for (uint32_t i = 0; i < num_blocks; ++i) {
    data.block_ids[i] = static_cast<int>(read32());
  }

  size_t payload_len = static_cast<size_t>(end - p);
  data.payload.assign(p, end);

  return data;
}

}  // namespace llm_engine
