// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>

#include "transport/kv_cache_layout.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Device-DRAM I/O addressed by (LocalDeviceId, NocAddr).
 *
 * The migrators depend on this interface rather than a concrete device so the
 * data plane unit-tests against an in-memory fake. DriscDeviceIo implements it
 * over the device's DRISC NOC-DMA path; the receiver drains and the sender
 * reads through it.
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

  // --- Async device I/O (optional; default is synchronous) -----------------
  // The overlap interface: a backend that can DMA in the background
  // (DRISC) issues transfers without blocking, so the caller can do other work
  // (stage the next window, drive the network) while the device DMA runs.
  //
  // Backends that can't (MMIO UMD, host/test fakes) inherit the default: the op
  // runs synchronously and is "already complete", so callers written against
  // the async interface behave identically (and correctly) on them.

  /// Issue a read without blocking; bytes land in `hostBuffer` by the time a
  /// later tryPopCompleted() retires it. `hostBuffer` must stay valid until
  /// then. @return false if it can't be issued right now (backpressure — the
  /// caller should tryPopCompleted() to retire an in-flight op, then retry).
  /// Default: run synchronously (completes inline).
  virtual bool readAsync(LocalDeviceId device, NocAddr nocAddr,
                         std::size_t size, void* hostBuffer) {
    return read(device, nocAddr, size, hostBuffer);
  }

  /// Issue a write without blocking (see readAsync). Default: synchronous.
  virtual bool writeAsync(LocalDeviceId device, NocAddr nocAddr,
                          const void* hostBuffer, std::size_t size) {
    return write(device, nocAddr, hostBuffer, size);
  }

  /// Retire one completed async op (a read copies its bytes into the caller's
  /// buffer). @return true if one was retired. Default: nothing outstanding.
  virtual bool tryPopCompleted() { return false; }

  /// Async ops issued but not yet retired. Default 0 (synchronous backend).
  virtual uint32_t asyncInFlight() const { return 0; }
};

}  // namespace tt::transport
