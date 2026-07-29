// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>

#include "transport/kv_cache_layout.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Device-DRAM I/O addressed by (LocalDeviceId, NocAddr).
 *
 * The migrators depend on this interface rather than a concrete device so the
 * data plane unit-tests against an in-memory fake. MultiDeviceUmd implements it
 * over real UMD; the receiver drains and the sender reads through it.
 */
class IDeviceIo {
 public:
  virtual ~IDeviceIo() = default;

  /// Read `size` bytes from `device` DRAM at `noc_addr` into `hostBuffer`.
  virtual bool read(LocalDeviceId device, NocAddr nocAddr, std::size_t size,
                    void* hostBuffer) = 0;

  /// Write `size` bytes from `hostBuffer` into `device` DRAM at `noc_addr`.
  virtual bool write(LocalDeviceId device, NocAddr nocAddr,
                     const void* hostBuffer, std::size_t size) = 0;
};

}  // namespace tt::transport
