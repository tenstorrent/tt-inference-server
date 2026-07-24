// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_bounce_buffer.hpp"

#include <cstdlib>
#include <string>

namespace tt::transport {

namespace {
// Read an unsigned tunable from the environment, falling back to `fallback` on
// unset/empty/unparseable/zero (matches the sender's staging tunables).
uint64_t envU64(const char* key, uint64_t fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return fallback;
  try {
    const unsigned long long parsed = std::stoull(v);
    return parsed == 0 ? fallback : static_cast<uint64_t>(parsed);
  } catch (...) {
    return fallback;
  }
}
}  // namespace

BounceGeometry defaultBounceGeometry() {
  BounceGeometry g;
  g.section_count = static_cast<uint32_t>(envU64(
      "TT_MOONCAKE_BOUNCE_SECTION_COUNT", K_DEFAULT_BOUNCE_SECTION_COUNT));
  g.section_size =
      envU64("TT_MOONCAKE_BOUNCE_SECTION_SIZE", K_DEFAULT_BOUNCE_SECTION_SIZE);
  return g;
}

KvBounceBuffer::KvBounceBuffer(BounceGeometry geometry) : geo_(geometry) {
  const uint64_t bytes = geo_.capacity();
  // Page-aligned (kAlign), uninitialized: only sections a migration actually
  // writes are read back, so untouched bytes never reach a device.
  // aligned_alloc needs the size to be a multiple of the alignment. Page
  // alignment lets the bounce buffer be NOC-mapped for DRISC DMA.
  constexpr uint64_t kAlign = 4096;
  if (bytes > 0) {
    const uint64_t rounded = (bytes + kAlign - 1) & ~(kAlign - 1);
    buffer_.reset(static_cast<uint8_t*>(std::aligned_alloc(kAlign, rounded)));
  }
}

uint8_t* KvBounceBuffer::sectionPtr(uint64_t offset, uint64_t size) {
  if (!buffer_) return nullptr;
  const uint64_t cap = geo_.capacity();
  // Overflow-safe: reject offset past the bounce buffer and any size that would
  // run past its end (checked as cap - offset to avoid offset + size
  // overflowing).
  if (offset > cap || size > cap - offset) return nullptr;
  return buffer_.get() + offset;
}

const uint8_t* KvBounceBuffer::sectionPtr(uint64_t offset,
                                          uint64_t size) const {
  return const_cast<KvBounceBuffer*>(this)->sectionPtr(offset, size);
}

}  // namespace tt::transport
