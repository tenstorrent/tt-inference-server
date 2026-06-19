// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "transport/kv_cache_layout.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief The decode-host (receiver) mirror buffer.
 *
 * A contiguous host buffer that is a *physical image* of the receiver's KV
 * cache region: its layout is derived from the local KvChunkAddressTable so
 * that host byte `offset_of(device, noc_addr)` is the 1:1 image of the device
 * byte at `noc_addr` on `device`. The whole buffer is registered as one
 * Mooncake segment; the prefill host writes each chunk to its computed offset
 * and the receiver later drains touched chunks back to device DRAM via UMD.
 *
 * Built from the *full* local table (every chunk this host can hold) so the
 * offsets are stable across migrations.
 *
 * Registration with the transfer engine and the device drain are wired in a
 * later phase; this type owns the buffer and the offset arithmetic.
 */
class KvCacheMirror {
 public:
  KvCacheMirror() = default;

  /// Build the layout from the local table's chunks and allocate the buffer.
  explicit KvCacheMirror(const std::vector<KvChunkLocation>& chunks);

  const KvCacheLayout& layout() const { return layout_; }
  uint64_t totalBytes() const { return layout_.totalBytes(); }

  /// Base of the registered host buffer (nullptr if empty).
  uint8_t* base() { return buffer_.empty() ? nullptr : buffer_.data(); }
  const uint8_t* base() const {
    return buffer_.empty() ? nullptr : buffer_.data();
  }

  /// Byte offset of a device address within the mirror; see KvCacheLayout.
  std::optional<uint64_t> offsetOf(LocalDeviceId device,
                                   NocAddr nocAddr) const {
    return layout_.offsetOf(device, nocAddr);
  }

  /**
   * @brief Pointer into the mirror buffer for the chunk at (device, noc_addr).
   * @return pointer, or nullptr if the address is not covered by the layout.
   */
  uint8_t* chunkPtr(LocalDeviceId device, NocAddr nocAddr);
  const uint8_t* chunkPtr(LocalDeviceId device, NocAddr nocAddr) const;

 private:
  KvCacheLayout layout_;
  std::vector<uint8_t> buffer_;
};

}  // namespace tt::transport
