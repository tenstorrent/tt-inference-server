// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Local (per-node) device identifier.
 *
 * On a node the host process reaches every TT device by its UMD chip id. This
 * is the per-node identity used to key device I/O and the mirror layout; the
 * cross-node identity (mesh_id, chip_id) of the disaggregation layer's
 * FabricNodeId is resolved down to this when the real KvChunkAddressTable is
 * adapted in a later phase.
 */
using LocalDeviceId = uint32_t;

/**
 * @brief Physical location of a single KV-cache chunk in device DRAM.
 *
 * A lightweight, dependency-free mirror of the disaggregation layer's
 * KvCacheLocation: which device, the NocAddr (`channel << 32 | local_addr`),
 * and the chunk's byte size. The address layer operates purely on these so it
 * builds and unit-tests with no tt-metal / protobuf dependency.
 */
struct KvChunkLocation {
  LocalDeviceId device = 0;
  NocAddr noc_addr = 0;
  uint64_t size_bytes = 0;
};

/**
 * @brief The mirror sub-region covering one (device, DRAM channel) pair.
 *
 * The decode host's mirror buffer is a *physical image* of each device
 * channel's KV span: within a region, host byte `seg_base + (local - dev_base)`
 * is the 1:1 image of device byte `local` on that channel. `dev_base` is the
 * lowest channel-local address used by any KV chunk on the (device, channel),
 * and `size` spans up to the highest chunk end.
 */
struct ChannelRegion {
  LocalDeviceId device = 0;
  uint32_t channel = 0;
  uint64_t dev_base = 0;  ///< Lowest channel-local addr covered by this region.
  uint64_t size = 0;      ///< Bytes covered (dev_base .. dev_base+size).
  uint64_t seg_base = 0;  ///< Byte offset of this region within the mirror.
};

/**
 * @brief The physical mirror layout for a KV-cache address table.
 *
 * Built deterministically from a set of KvChunkLocations: chunks are grouped by
 * (device, channel), each group becomes a ChannelRegion sized to span its
 * chunks, and regions are packed back-to-back in a stable (device, channel)
 * order. The resulting `offset_of()` maps any device NocAddr to its byte offset
 * inside one contiguous mirror buffer / Mooncake segment.
 *
 * The same constructor runs on both sides — the decode host builds a
 * KvCacheMirror over it, the prefill host builds a RemoteRegion over the
 * exchanged table — so an offset computed by the sender is byte-identical to
 * the one the receiver drains from. Parity is structural, not coincidental.
 *
 * Build the layout from the *full* table (every chunk the receiver can hold),
 * not a single migration's subset, so offsets stay stable across migrations.
 */
class KvCacheLayout {
 public:
  KvCacheLayout() = default;
  explicit KvCacheLayout(const std::vector<KvChunkLocation>& chunks);

  /// Total bytes of the mirror buffer / segment this layout describes.
  uint64_t totalBytes() const { return total_bytes_; }

  std::size_t numRegions() const { return regions_.size(); }
  const std::vector<ChannelRegion>& regions() const { return regions_; }

  /**
   * @brief Byte offset of `noc_addr` on `device` within the mirror / segment.
   * @return the offset, or std::nullopt if the (device, channel) is not part of
   *         this layout or the address falls outside its region.
   */
  std::optional<uint64_t> offsetOf(LocalDeviceId device, NocAddr nocAddr) const;

 private:
  /// Packs (device, channel) into the region-index map key.
  static uint64_t regionKey(LocalDeviceId device, uint32_t channel) {
    return (static_cast<uint64_t>(device) << 32) | channel;
  }

  std::vector<ChannelRegion> regions_;  ///< Sorted by (device, channel).
  std::map<uint64_t, std::size_t>
      index_;  ///< regionKey -> index into regions_.
  uint64_t total_bytes_ = 0;
};

}  // namespace tt::transport
