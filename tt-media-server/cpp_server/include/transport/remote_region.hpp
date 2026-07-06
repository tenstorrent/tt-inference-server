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
 * @brief The prefill-host (sender) view of the decode host's KV region.
 *
 * Built at init from the receiver's exchanged address table plus the opened
 * Mooncake segment handle. It owns the *destination* layout, so the sender
 * computes the full destination addressing on its own side:
 *
 *   - mirror_offset(device, noc_addr) — the Mooncake `target_offset` for a
 *     one-sided WRITE into the receiver's mirror segment (the TCP path today);
 *   - the destination device NocAddr itself is already known to the sender from
 *     the same table, so a future device-to-device RDMA-direct path writes
 *     straight to the device with no change to addressing.
 *
 * Because the layout is built by the same KvCacheLayout constructor the
 * receiver uses for its KvCacheMirror, mirror_offset() is byte-identical to the
 * receiver's offset_of() for the same address.
 */
class RemoteRegion {
 public:
  RemoteRegion() = default;

  /// @param segment    handle returned by ITransferEngine::openSegment.
  /// @param dst_chunks the receiver's full table, as exchanged at init.
  RemoteRegion(SegmentHandle segment,
               const std::vector<KvChunkLocation>& dstChunks);

  SegmentHandle segment() const { return segment_; }
  const KvCacheLayout& layout() const { return layout_; }
  uint64_t totalBytes() const { return layout_.totalBytes(); }

  /**
   * @brief Mooncake target_offset for a WRITE of the chunk at (device,
   *        noc_addr) on the receiver. Identical to the receiver's
   *        KvCacheMirror::offset_of for the same address.
   * @return offset, or std::nullopt if the address is not in the layout.
   */
  std::optional<uint64_t> mirrorOffset(LocalDeviceId device,
                                       NocAddr nocAddr) const {
    return layout_.offsetOf(device, nocAddr);
  }

 private:
  SegmentHandle segment_ = K_INVALID_SEGMENT;
  KvCacheLayout layout_;
};

}  // namespace tt::transport
