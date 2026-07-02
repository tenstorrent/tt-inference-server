// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "transport/i_storage_backend.hpp"
#include "transport/transfer_types.hpp"
#include "transport/umd_device_access.hpp"

namespace tt::transport {

/**
 * @brief Device-DRAM storage backend via the UMD — the custom backend #3890
 *        targets ("Custom backend for DRAM access via UMD").
 *
 * The backing store is TT device DRAM; `addr` is a NocAddr
 * (`channel << 32 | local_addr`). Delegates the actual device I/O to
 * UmdDeviceAccess, so Mooncake's transport never touches device DRAM directly —
 * bytes are staged through the registered host buffer (see
 * mooncake/poc-transfer-engine/adr-mooncake-backend.md, storage/transport
 * split). The UMD I/O underneath is real when built with tt-metal
 * (USE_METAL_CPP_LIB); otherwise UmdDeviceAccess reports failure.
 */
class DeviceDramStorageBackend : public IStorageBackend {
 public:
  explicit DeviceDramStorageBackend(std::shared_ptr<UmdDeviceAccess> device);

  StorageMedium medium() const override { return StorageMedium::DEVICE_DRAM; }

  bool readInto(uint64_t addr, std::size_t size, void* hostBuffer) override;
  bool writeFrom(uint64_t addr, const void* hostBuffer,
                 std::size_t size) override;

 private:
  std::shared_ptr<UmdDeviceAccess> device_;
};

}  // namespace tt::transport
