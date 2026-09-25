// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "services/embedding_pipe.hpp"

#include <unistd.h>

#include <cstring>

#include "config/defaults.hpp"

namespace tt::services::embedding_detail {

bool pipeWrite(int fd, const void* data, size_t len) {
  uint32_t header = static_cast<uint32_t>(len);
  if (write(fd, &header, sizeof(header)) != sizeof(header)) return false;
  return write(fd, data, len) == static_cast<ssize_t>(len);
}

std::vector<uint8_t> pipeReadBinary(int fd) {
  uint32_t len = 0;
  ssize_t n = read(fd, &len, sizeof(len));
  if (n != sizeof(len) || len > tt::config::defaults::EMBEDDING_MAX_PIPE_BYTES)
    return {};

  std::vector<uint8_t> buf(len);
  size_t total = 0;
  while (total < len) {
    n = read(fd, buf.data() + total, len - total);
    if (n <= 0) return {};
    total += static_cast<size_t>(n);
  }
  return buf;
}

std::string pipeReadString(int fd) {
  uint32_t len = 0;
  ssize_t n = read(fd, &len, sizeof(len));
  if (n != sizeof(len) || len > tt::config::defaults::EMBEDDING_MAX_PIPE_BYTES)
    return {};

  std::string data(len, '\0');
  size_t total = 0;
  while (total < len) {
    n = read(fd, data.data() + total, len - total);
    if (n <= 0) return {};
    total += static_cast<size_t>(n);
  }
  return data;
}

bool isReadySentinel(const std::vector<uint8_t>& payload) {
  constexpr size_t sentinelLen = sizeof(WORKER_READY_SENTINEL) - 1;
  return payload.size() == sentinelLen &&
         std::memcmp(payload.data(), WORKER_READY_SENTINEL, sentinelLen) == 0;
}

}  // namespace tt::services::embedding_detail
