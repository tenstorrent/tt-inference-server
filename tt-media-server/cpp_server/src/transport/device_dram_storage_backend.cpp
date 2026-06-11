// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/device_dram_storage_backend.hpp"

#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

DeviceDramStorageBackend::DeviceDramStorageBackend(
    std::shared_ptr<UmdDeviceAccess> device)
    : device_(std::move(device)) {}

bool DeviceDramStorageBackend::readInto(uint64_t addr, std::size_t size,
                                        void* hostBuffer) {
  if (!device_) {
    TT_LOG_ERROR("[DeviceDramStorageBackend] readInto with no UmdDeviceAccess");
    return false;
  }
  // The custom backend is just the UMD device-DRAM read, keyed by NocAddr.
  return device_->read(static_cast<NocAddr>(addr), size, hostBuffer);
}

bool DeviceDramStorageBackend::writeFrom(uint64_t addr, const void* hostBuffer,
                                         std::size_t size) {
  if (!device_) {
    TT_LOG_ERROR(
        "[DeviceDramStorageBackend] writeFrom with no UmdDeviceAccess");
    return false;
  }
  return device_->write(static_cast<NocAddr>(addr), hostBuffer, size);
}

}  // namespace tt::transport
