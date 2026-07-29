// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Enumerate the visible UMD devices once, mapping each chip's 64-bit
 *        ASIC `unique_id` to its 0-based `UmdDevice::open()` index.
 *
 * The DeviceMap keys a FabricNode to a chip's stable ASIC `unique_id` (the
 * third device-map column; see device_map.hpp / print_local_device_map). But
 * `UmdDeviceAccess`/`UmdDevice::open()` take a 0-based index into the sorted
 * set of visible MMIO chips, NOT a unique_id — casting the id to an index opens
 * the wrong chip (or aborts). This resolver bridges the two: enumerate
 * `count()` chips via `open(i)` + `unique_id()` a single time, so a caller can
 * translate a device-map unique_id into the index `open()` actually expects.
 * Mirrors the disaggregation worker's `open_all_devices`
 * (chip_index_by_unique).
 *
 * @note Enumerating opens each visible chip once (the shared UMD Cluster makes
 *       re-opening cheap); call it ONCE and reuse the map, never per device.
 * @note The no-tt-metal fallback build returns an empty map (there are no real
 *       devices to open); a non-empty DeviceMap then resolves to a clean miss.
 */
std::unordered_map<uint64_t, int> enumerateUmdDevicesByUniqueId();

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
  explicit UmdDeviceAccess(int deviceId = 0);
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

  /**
   * @brief Number of DRAM channels (Metal dram_views) on the opened device.
   * @return the channel count, or 0 if the device failed to open or tt-metal is
   *         not in this build. A NocAddr's channel must be `<
   * numDramChannels()`.
   */
  uint32_t numDramChannels() const;

 private:
  // Hides tt-metal's UmdDevice from this header.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::transport
