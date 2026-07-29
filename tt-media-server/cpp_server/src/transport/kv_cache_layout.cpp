// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_cache_layout.hpp"

#include <algorithm>
#include <utility>

namespace tt::transport {

KvCacheLayout::KvCacheLayout(const std::vector<KvChunkLocation>& chunks) {
  // 1. Per (device, channel), find the min channel-local base and max chunk
  //    end. std::map keeps the keys in a stable (device, channel) order so the
  //    packing below is deterministic and identical on both sides.
  struct Span {
    uint64_t base = 0;
    uint64_t end = 0;
    bool seen = false;
  };
  std::map<uint64_t, Span> spans;
  for (const auto& chunk : chunks) {
    if (chunk.size_bytes == 0) continue;  // empty chunk contributes no span
    const uint32_t channel = nocChannel(chunk.noc_addr);
    const uint64_t local = nocLocalAddr(chunk.noc_addr);
    const uint64_t end = local + chunk.size_bytes;
    Span& span = spans[regionKey(chunk.device, channel)];
    if (!span.seen) {
      span.base = local;
      span.end = end;
      span.seen = true;
    } else {
      span.base = std::min(span.base, local);
      span.end = std::max(span.end, end);
    }
  }

  // 2. Pack each (device, channel) span into a back-to-back region, assigning a
  //    running segment offset. The host mirror byte at seg_base + (local-base)
  //    is the 1:1 image of device byte `local` on that channel.
  uint64_t running = 0;
  regions_.reserve(spans.size());
  for (const auto& [key, span] : spans) {
    ChannelRegion region;
    region.device = static_cast<LocalDeviceId>(key >> 32);
    region.channel = static_cast<uint32_t>(key & 0xFFFFFFFFULL);
    region.dev_base = span.base;
    region.size = span.end - span.base;
    region.seg_base = running;
    running += region.size;
    index_.emplace(key, regions_.size());
    regions_.push_back(region);
  }
  total_bytes_ = running;
}

std::optional<uint64_t> KvCacheLayout::offsetOf(LocalDeviceId device,
                                                NocAddr nocAddr) const {
  const uint32_t channel = nocChannel(nocAddr);
  const auto it = index_.find(regionKey(device, channel));
  if (it == index_.end()) return std::nullopt;

  const ChannelRegion& region = regions_[it->second];
  const uint64_t local = nocLocalAddr(nocAddr);
  if (local < region.dev_base) return std::nullopt;
  const uint64_t offsetInRegion = local - region.dev_base;
  if (offsetInRegion >= region.size) return std::nullopt;
  return region.seg_base + offsetInRegion;
}

}  // namespace tt::transport
