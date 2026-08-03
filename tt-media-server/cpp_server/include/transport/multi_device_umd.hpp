// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "transport/i_device_io.hpp"
#include "transport/kv_cache_layout.hpp"
#include "transport/transfer_types.hpp"
#include "transport/umd_device_access.hpp"

namespace tt::transport {

/**
 * @brief Device-DRAM I/O across the multiple TT devices on one node.
 *
 * A whole-slot migration touches several devices on a node (layers spread
 * across devices). This holds one UmdDeviceAccess per LocalDeviceId and
 * dispatches reads/writes by (device, NocAddr) — the sender reads each chunk
 * from its source device, the receiver drains each chunk to its destination
 * device.
 *
 * It composes the existing UmdDeviceAccess, so it inherits the same build
 * guard: real device DRAM I/O with USE_METAL_CPP_LIB, a no-op that reports
 * failure otherwise (keeping transport_lib buildable in every configuration).
 */
class MultiDeviceUmd : public IDeviceIo {
 public:
  MultiDeviceUmd() = default;

  /// Register the UMD handle for a device. Replaces any existing handle.
  void addDevice(LocalDeviceId device, std::shared_ptr<UmdDeviceAccess> access);

  /// True if a handle is registered for `device`.
  bool hasDevice(LocalDeviceId device) const;

  std::size_t numDevices() const { return devices_.size(); }

  /**
   * @brief Read `size` bytes from `device` DRAM at `noc_addr` into
   * `hostBuffer`.
   * @return true on success; false if the device is unknown or the I/O fails.
   */
  bool read(LocalDeviceId device, NocAddr nocAddr, std::size_t size,
            void* hostBuffer) override;

  /**
   * @brief Write `size` bytes from `hostBuffer` into `device` DRAM at
   *        `noc_addr`.
   * @return true on success; false if the device is unknown or the I/O fails.
   */
  bool write(LocalDeviceId device, NocAddr nocAddr, const void* hostBuffer,
             std::size_t size) override;

 private:
  std::unordered_map<LocalDeviceId, std::shared_ptr<UmdDeviceAccess>> devices_;
};

}  // namespace tt::transport
