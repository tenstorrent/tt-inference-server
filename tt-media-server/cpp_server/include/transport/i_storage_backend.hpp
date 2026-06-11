// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>

#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief The storage mechanism of the Transfer Engine — and the interface a
 *        *custom backend* plugs into (issue #3890, "Custom backend for DRAM
 *        access via UMD").
 *
 * Per #3890's core assumption, the Transfer Engine defines both a *storage*
 * mechanism (host DRAM, device DRAM, ...) and a *transport* mechanism (TCP,
 * RDMA, ...). This interface abstracts the storage mechanism: it stages bytes
 * between a backing store and a host staging buffer that the transport then
 * moves between galaxies.
 *
 * Implementations:
 *   - HostDramStorageBackend   — plain host memory.
 *   - DeviceDramStorageBackend — TT device DRAM via the UMD (the custom backend
 *                                #3890 targets).
 *
 * `addr` is interpreted by the concrete backend: a NocAddr
 * (`channel << 32 | local_addr`) for device DRAM, or a host virtual address
 * for host DRAM.
 */
class IStorageBackend {
 public:
  virtual ~IStorageBackend() = default;

  /// Which storage mechanism this backend implements.
  virtual StorageMedium medium() const = 0;

  /**
   * @brief Copy `size` bytes from the backing store at `addr` into
   *        `hostBuffer` (the registered staging buffer). The "read" side of a
   *        sender-galaxy transfer.
   * @return true on success.
   */
  virtual bool readInto(uint64_t addr, std::size_t size, void* hostBuffer) = 0;

  /**
   * @brief Copy `size` bytes from `hostBuffer` into the backing store at
   *        `addr`. The "write" side of a receiver-galaxy transfer.
   * @return true on success.
   */
  virtual bool writeFrom(uint64_t addr, const void* hostBuffer,
                         std::size_t size) = 0;
};

}  // namespace tt::transport
