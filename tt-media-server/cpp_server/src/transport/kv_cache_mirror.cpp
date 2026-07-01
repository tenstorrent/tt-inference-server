// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_cache_mirror.hpp"

namespace tt::transport {

KvCacheMirror::KvCacheMirror(const std::vector<KvChunkLocation>& chunks)
    : layout_(chunks), buffer_(layout_.totalBytes(), 0) {}

uint8_t* KvCacheMirror::chunkPtr(LocalDeviceId device, NocAddr nocAddr) {
  const auto offset = layout_.offsetOf(device, nocAddr);
  if (!offset || buffer_.empty()) return nullptr;
  return buffer_.data() + *offset;
}

const uint8_t* KvCacheMirror::chunkPtr(LocalDeviceId device,
                                       NocAddr nocAddr) const {
  const auto offset = layout_.offsetOf(device, nocAddr);
  if (!offset || buffer_.empty()) return nullptr;
  return buffer_.data() + *offset;
}

}  // namespace tt::transport
