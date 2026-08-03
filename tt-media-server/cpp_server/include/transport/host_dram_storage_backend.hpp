// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>

#include "transport/i_storage_backend.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Host-DRAM storage backend: the backing store is plain host memory.
 *
 * The trivial half of the storage mechanism (#3890) — `addr` is a host virtual
 * address, so staging to/from the registered host buffer is a plain memcpy.
 * Useful as the transport-only baseline (no device, no UMD) against which the
 * device-DRAM custom backend is compared.
 */
class HostDramStorageBackend : public IStorageBackend {
 public:
  StorageMedium medium() const override { return StorageMedium::HOST_DRAM; }

  bool readInto(uint64_t addr, std::size_t size, void* hostBuffer) override;
  bool writeFrom(uint64_t addr, const void* hostBuffer,
                 std::size_t size) override;
};

}  // namespace tt::transport
