// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>

namespace tt::transport {

/// Default number of sections in a bounce buffer.
inline constexpr uint32_t K_DEFAULT_BOUNCE_SECTION_COUNT = 4;
/// Default size of ONE bounce section in bytes (4 x 8 MiB = 32 MiB bounce
/// buffer by default).
inline constexpr uint64_t K_DEFAULT_BOUNCE_SECTION_SIZE = 8ull * 1024 * 1024;

/**
 * @brief Geometry of a bounce buffer: `section_count` fixed-size sections.
 *
 * The receiver registers a small bounce buffer of `section_count *
 * section_size` bytes and a migration streams through it a window at a time:
 * the sender fills free sections, the receiver drains them to device and
 * returns the sections as credits.
 */
struct BounceGeometry {
  uint32_t section_count = 0;
  uint64_t section_size = 0;

  uint64_t capacity() const {
    return static_cast<uint64_t>(section_count) * section_size;
  }
  bool valid() const { return section_count > 0 && section_size > 0; }
  bool operator==(const BounceGeometry& o) const {
    return section_count == o.section_count && section_size == o.section_size;
  }
};

/// The default bounce geometry, overridable by env
/// (`TT_MOONCAKE_BOUNCE_SECTION_COUNT`, `TT_MOONCAKE_BOUNCE_SECTION_SIZE`).
BounceGeometry defaultBounceGeometry();

/**
 * @brief Sender-side section allocator with credit accounting.
 *
 * Hands out section byte-offsets into a bounce buffer of the given geometry and
 * tracks how many are outstanding (filled but not yet drained). alloc() returns
 * nullopt once every section is outstanding — the sender must then flush the
 * window (send it, wait for the receiver to drain + ack) and release() the
 * freed credits before allocating again. Because a window is drained in full
 * before the next begins, offsets restart at 0 each window; a genuinely
 * pipelined bounce buffer (sections freed out of order) is a later optimization
 * and would need a real free-list.
 */
class BounceSectionAllocator {
 public:
  explicit BounceSectionAllocator(BounceGeometry geometry) : geo_(geometry) {}

  /// Byte offset of the next free section, or nullopt if all sections are
  /// outstanding.
  std::optional<uint64_t> alloc() {
    if (geo_.section_count == 0 || outstanding_ >= geo_.section_count) {
      return std::nullopt;
    }
    const uint64_t offset =
        static_cast<uint64_t>(outstanding_) * geo_.section_size;
    ++outstanding_;
    return offset;
  }

  /// Return `n` drained sections to the free pool (clamped at outstanding).
  void release(uint32_t n) {
    outstanding_ = (n >= outstanding_) ? 0 : (outstanding_ - n);
  }

  uint32_t outstanding() const { return outstanding_; }
  uint32_t freeSections() const { return geo_.section_count - outstanding_; }
  const BounceGeometry& geometry() const { return geo_; }

 private:
  BounceGeometry geo_;
  uint32_t outstanding_ = 0;
};

/**
 * @brief The receiver-side bounce buffer: a small registered host buffer.
 *
 * Owns `capacity()` bytes registered with the transfer engine as the one
 * Mooncake segment the sender writes into. It is small and dense, so
 * `ibv_reg_mr` pins only tens of MiB — the whole point of RDMA-over-host. The
 * bounce buffer carries no addressing itself: each window's descriptors (sent
 * over the control channel) tell the receiver which device address every
 * section drains to, so the sender owns all destination addressing and the
 * receiver holds no table.
 */
class KvBounceBuffer {
 public:
  KvBounceBuffer() = default;

  /// Allocate `geometry.capacity()` bytes (uninitialized — only sections a
  /// migration writes are ever read back).
  explicit KvBounceBuffer(BounceGeometry geometry);

  const BounceGeometry& geometry() const { return geo_; }
  uint64_t totalBytes() const { return geo_.capacity(); }

  uint8_t* base() { return buffer_.get(); }
  const uint8_t* base() const { return buffer_.get(); }

  /// Pointer to `[offset, offset+size)` within the bounce buffer, or nullptr if
  /// that range escapes the bounce buffer (the essential bounds check on an
  /// untrusted sender-supplied descriptor).
  uint8_t* sectionPtr(uint64_t offset, uint64_t size);
  const uint8_t* sectionPtr(uint64_t offset, uint64_t size) const;

 private:
  /// Page-aligned so the bounce buffer can be NOC-mapped for DRISC DMA (>= the
  /// 64-byte DRISC payload contract) and RDMA-registered as one arena.
  /// `std::free` since the buffer comes from `std::aligned_alloc`.
  struct AlignedFree {
    void operator()(uint8_t* p) const { std::free(p); }
  };

  BounceGeometry geo_;
  std::unique_ptr<uint8_t[], AlignedFree> buffer_;
};

}  // namespace tt::transport
