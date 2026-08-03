// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_cache_mirror.hpp"

namespace tt::transport {

KvCacheMirror::KvCacheMirror(const std::vector<KvChunkLocation>& chunks)
    : layout_(chunks) {
  const uint64_t bytes = layout_.totalBytes();
  // `new uint8_t[]` default-initializes: for a trivial type the bytes are left
  // UNinitialized (no zero-fill). This is deliberate — see the buffer_ comment.
  if (bytes > 0) buffer_.reset(new uint8_t[bytes]);
}

uint8_t* KvCacheMirror::chunkPtr(LocalDeviceId device, NocAddr nocAddr) {
  const auto offset = layout_.offsetOf(device, nocAddr);
  if (!offset || !buffer_) return nullptr;
  return buffer_.get() + *offset;
}

const uint8_t* KvCacheMirror::chunkPtr(LocalDeviceId device,
                                       NocAddr nocAddr) const {
  const auto offset = layout_.offsetOf(device, nocAddr);
  if (!offset || !buffer_) return nullptr;
  return buffer_.get() + *offset;
}

}  // namespace tt::transport
