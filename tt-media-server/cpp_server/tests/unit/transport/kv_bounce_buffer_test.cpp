// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_bounce_buffer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "transport/kv_control_message.hpp"

namespace tt::transport {
namespace {

// The allocator hands out one offset per slot, then backpressures until credits
// are released, and offsets restart at 0 after a full-window release.
TEST(BounceSectionAllocator, HandsOutSlotsThenBackpressures) {
  BounceSectionAllocator alloc(
      BounceGeometry{/*section_count=*/3, /*section_size=*/1024});
  EXPECT_EQ(alloc.freeSections(), 3u);

  const auto a = alloc.alloc();
  const auto b = alloc.alloc();
  const auto c = alloc.alloc();
  ASSERT_TRUE(a && b && c);
  EXPECT_EQ(*a, 0u);
  EXPECT_EQ(*b, 1024u);
  EXPECT_EQ(*c, 2048u);
  EXPECT_EQ(alloc.outstanding(), 3u);

  // Full: no slot until credits come back.
  EXPECT_FALSE(alloc.alloc().has_value());

  // Release the whole window; offsets restart at 0.
  alloc.release(3);
  EXPECT_EQ(alloc.outstanding(), 0u);
  const auto d = alloc.alloc();
  ASSERT_TRUE(d);
  EXPECT_EQ(*d, 0u);
}

// release() clamps and a zero-slot geometry never allocates.
TEST(BounceSectionAllocator, ReleaseClampsAndEmptyGeometry) {
  BounceSectionAllocator alloc(BounceGeometry{2, 512});
  alloc.alloc();
  alloc.release(10);  // more than outstanding -> clamps to 0
  EXPECT_EQ(alloc.outstanding(), 0u);

  BounceSectionAllocator empty(BounceGeometry{0, 512});
  EXPECT_FALSE(empty.alloc().has_value());
}

// sectionPtr admits in-bounds ranges and rejects anything running past the
// bounce buffer, including overflow-crafted offsets.
TEST(KvBounceBuffer, SlotPtrBoundsCheck) {
  KvBounceBuffer buf(BounceGeometry{/*section_count=*/2, /*section_size=*/64});
  ASSERT_NE(buf.base(), nullptr);
  EXPECT_EQ(buf.totalBytes(), 128u);

  EXPECT_EQ(buf.sectionPtr(0, 64), buf.base());
  EXPECT_EQ(buf.sectionPtr(64, 64), buf.base() + 64);
  EXPECT_EQ(buf.sectionPtr(0, 128), buf.base());  // whole bounce buffer

  EXPECT_EQ(buf.sectionPtr(64, 65), nullptr);  // runs one byte past
  EXPECT_EQ(buf.sectionPtr(129, 1), nullptr);  // offset past the bounce buffer
  EXPECT_EQ(buf.sectionPtr(0, ~0ull), nullptr);  // overflow-safe reject
}

// An empty bounce buffer hands out no usable pointer.
TEST(KvBounceBuffer, EmptyBufferIsInert) {
  KvBounceBuffer buf;
  EXPECT_EQ(buf.base(), nullptr);
  EXPECT_EQ(buf.totalBytes(), 0u);
  EXPECT_EQ(buf.sectionPtr(0, 1), nullptr);
}

// The new bounce-path control messages survive a serialize/deserialize
// round-trip, including the nested window descriptors with fan-out targets.
TEST(KvControlMessage, RoundTripsBounceKinds) {
  KvControlMessage bounceReady;
  bounceReady.type = KvControlType::BOUNCE_READY;
  bounceReady.uuid = 0x1234;
  bounceReady.segment_name = "decode-A:127.0.0.1:9000";
  bounceReady.bounce_section_count = 4;
  bounceReady.bounce_section_size = 8ull * 1024 * 1024;

  KvControlMessage windowReady;
  windowReady.type = KvControlType::WINDOW_READY;
  windowReady.uuid = 0x1234;
  windowReady.window = {
      BounceSectionDescriptor{0, 19584, {{0x20000, 0x100}, {0x20001, 0x100}}},
      BounceSectionDescriptor{
          8ull * 1024 * 1024, 64, {{0x30000, 0x2000000000ull}}},
      BounceSectionDescriptor{
          16ull * 1024 * 1024, 0, {}},  // degenerate but valid
  };

  KvControlMessage windowAck;
  windowAck.type = KvControlType::WINDOW_ACK;
  windowAck.uuid = 0x1234;
  windowAck.credits = 3;
  windowAck.ok = true;

  for (const auto& m : {bounceReady, windowReady, windowAck}) {
    const auto bytes = m.serialize();
    const auto parsed = KvControlMessage::deserialize(bytes);
    ASSERT_TRUE(parsed.has_value());
    EXPECT_TRUE(*parsed == m);
  }
}

// A truncated window descriptor is rejected, not misparsed.
TEST(KvControlMessage, RejectsTruncatedWindow) {
  KvControlMessage m;
  m.type = KvControlType::WINDOW_READY;
  m.window = {BounceSectionDescriptor{0, 128, {{7, 0x1000}}}};
  auto bytes = m.serialize();
  bytes.resize(bytes.size() - 1);  // drop the last target byte
  EXPECT_FALSE(KvControlMessage::deserialize(bytes).has_value());
}

}  // namespace
}  // namespace tt::transport
