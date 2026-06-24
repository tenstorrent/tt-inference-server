// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Access wrapper for TT device DRAM via the UMD (User-Mode Driver).
 *
 * This is the *storage* half of the storage/transport split (see
 * mooncake/poc-transfer-engine/adr-mooncake-backend.md): it stages bytes
 * between a host buffer and TT device DRAM, addressed by NocAddr (`channel <<
 * 32 | local_addr`). The transport layer (ITransferEngine) then moves the host
 * buffer between hosts.
 *
 * The bounce-buffer migration path is:
 * @code
 *   sender DRAM --[read(noc_addr, ..., host_buf)]--> host_buf (registered)
 *       --[ITransferEngine::submitAndWait]--> peer host_buf
 *       --[write(noc_addr, host_buf, ...)]--> receiver DRAM
 * @endcode
 *
 * Backed by tt-metal's device handle (`IDevice`), hidden behind a pimpl so this
 * header has no tt-metal dependency; the real device handle is pulled in only
 * by the .cpp.
 *
 * @note The .cpp has two implementations selected at build time: the real UMD
 *       backend (tt_metal Read/WriteFromDeviceDRAMChannel) when built with
 *       `USE_METAL_CPP_LIB`, and a no-op fallback that reports failure
 * otherwise, so the transport library still builds without tt-metal (#3890).
 */
class UmdDeviceAccess {
 public:
  /**
   * @brief Open the UMD device with the given chip/device id.
   */
  explicit UmdDeviceAccess(int device_id = 0);
  ~UmdDeviceAccess();

  UmdDeviceAccess(const UmdDeviceAccess&) = delete;
  UmdDeviceAccess& operator=(const UmdDeviceAccess&) = delete;
  UmdDeviceAccess(UmdDeviceAccess&&) noexcept;
  UmdDeviceAccess& operator=(UmdDeviceAccess&&) noexcept;

  /**
   * @brief Read `size` bytes from device DRAM at `addr` into `hostBuffer`.
   * @return true on success.
   */
  bool read(NocAddr addr, std::size_t size, void* hostBuffer);

  /**
   * @brief Write `size` bytes from `hostBuffer` into device DRAM at `addr`.
   * @return true on success.
   */
  bool write(NocAddr addr, const void* hostBuffer, std::size_t size);

 private:
  // Hides tt-metal's UmdDevice from this header.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::transport
