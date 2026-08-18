// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/host_dram_storage_backend.hpp"

#include <cstring>

#include "utils/logger.hpp"

namespace tt::transport {

// For the host-DRAM backend the "backing store" is plain host memory, so `addr`
// is a host virtual address and staging is a straight memcpy between it and the
// registered host buffer.

bool HostDramStorageBackend::readInto(uint64_t addr, std::size_t size,
                                      void* hostBuffer) {
  if (size == 0) {
    return true;
  }
  if (addr == 0 || hostBuffer == nullptr) {
    TT_LOG_ERROR(
        "[HostDramStorageBackend] readInto(addr={:#x}, size={}) with null "
        "addr or hostBuffer",
        addr, size);
    return false;
  }
  std::memcpy(hostBuffer, reinterpret_cast<const void*>(addr), size);
  return true;
}

bool HostDramStorageBackend::writeFrom(uint64_t addr, const void* hostBuffer,
                                       std::size_t size) {
  if (size == 0) {
    return true;
  }
  if (addr == 0 || hostBuffer == nullptr) {
    TT_LOG_ERROR(
        "[HostDramStorageBackend] writeFrom(addr={:#x}, size={}) with null "
        "addr or hostBuffer",
        addr, size);
    return false;
  }
  std::memcpy(reinterpret_cast<void*>(addr), hostBuffer, size);
  return true;
}

}  // namespace tt::transport
