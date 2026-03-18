// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "runners/kv_cache_migrator.hpp"

namespace llm_engine {

/**
 * H2H (host-to-host) TCP socket implementation of IKVCacheMigrator.
 *
 * Mocks device-to-device KV cache transfer using a length-prefixed binary
 * protocol over a single persistent TCP connection.
 *
 * Wire format per message:
 *   [4B total_len (network order)]
 *   [4B task_id_len][task_id bytes]
 *   [4B num_blocks][block_id_0 ... block_id_N-1  (4B each)]
 *   [remaining bytes: raw tensor payload]
 */
class H2HKVCacheMigrator : public IKVCacheMigrator {
 public:
  enum class Mode { SERVER, CLIENT };

  H2HKVCacheMigrator(Mode mode, const std::string& host, uint16_t port);
  ~H2HKVCacheMigrator() override;

  H2HKVCacheMigrator(const H2HKVCacheMigrator&) = delete;
  H2HKVCacheMigrator& operator=(const H2HKVCacheMigrator&) = delete;

  void send(KVCacheMigrationData data) override;
  void setReceiveCallback(ReceiveCallback cb) override;
  void start() override;
  void stop() override;

 private:
  void serverLoop();
  void clientLoop();
  void receiveLoop();

  bool sendAll(int fd, const void* buf, size_t len);
  bool recvAll(int fd, void* buf, size_t len);

  std::vector<uint8_t> serialize(const KVCacheMigrationData& data);
  KVCacheMigrationData deserialize(const std::vector<uint8_t>& buf);

  Mode mode_;
  std::string host_;
  uint16_t port_;

  int server_fd_ = -1;
  int peer_fd_ = -1;

  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};
  std::mutex send_mutex_;

  std::thread connect_thread_;
  std::thread receive_thread_;

  ReceiveCallback receive_callback_;
};

}  // namespace llm_engine
