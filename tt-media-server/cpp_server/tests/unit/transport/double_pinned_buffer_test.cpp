// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/double_pinned_buffer.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "transport/kv_staging_pool.hpp"  // KvStagingPool
#include "transport_test_fakes.hpp"

namespace tt::transport {
namespace {

using test::FakeRegistry;
using test::FakeTransferEngine;

// Records every (va, bytes) the device registrar was asked to NOC-map — the
// stand-in for DriscDeviceIo::registerHostRegion.
struct RecordingDeviceMap {
  std::vector<std::pair<void*, std::size_t>> calls;
  DeviceMapFn fn() {
    return
        [this](void* va, std::size_t bytes) { calls.emplace_back(va, bytes); };
  }
};

// The buffer is engine-registered (ibv_reg_mr surrogate), page-aligned, and the
// device map is invoked with the SAME base + the (rounded-up) capacity — the
// one-buffer-two-pinnings contract.
TEST(DoublePinnedBuffer, RegistersEngineAndDeviceOnOneAlignedBuffer) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<FakeTransferEngine>(reg, "P");
  RecordingDeviceMap dev;

  {
    DoublePinnedBuffer buf(engine, /*bytes=*/1000, dev.fn());
    ASSERT_TRUE(buf.registered());
    ASSERT_NE(buf.base(), nullptr);
    EXPECT_EQ(buf.size(), 1000u);
    EXPECT_EQ(buf.capacity(), 4096u);  // rounded up to page
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(buf.base()) % DoublePinnedBuffer::kAlign,
        0u);

    // Engine registered the buffer at its base for the whole capacity.
    ASSERT_NE(reg->segs.find("P"), reg->segs.end());
    EXPECT_EQ(reg->segs["P"].first, buf.base());
    EXPECT_EQ(reg->segs["P"].second, buf.capacity());

    // Device map hit exactly once, same base + capacity.
    ASSERT_EQ(dev.calls.size(), 1u);
    EXPECT_EQ(dev.calls[0].first, buf.base());
    EXPECT_EQ(dev.calls[0].second, buf.capacity());
  }
  // RAII unregistered the engine side.
  EXPECT_EQ(reg->segs.find("P"), reg->segs.end());
}

// With no device map (host/MMIO mode) the buffer is engine-only — still valid,
// no NOC mapping attempted.
TEST(DoublePinnedBuffer, NoDeviceMapIsEngineOnly) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<FakeTransferEngine>(reg, "P");
  DoublePinnedBuffer buf(engine, 4096);  // default (null) device map
  EXPECT_TRUE(buf.registered());
  EXPECT_NE(reg->segs.find("P"), reg->segs.end());
}

// A zero-byte request allocates nothing and registers nothing (no crash).
TEST(DoublePinnedBuffer, ZeroBytesIsInert) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<FakeTransferEngine>(reg, "P");
  RecordingDeviceMap dev;
  DoublePinnedBuffer buf(engine, 0, dev.fn());
  EXPECT_FALSE(buf.registered());
  EXPECT_EQ(buf.base(), nullptr);
  EXPECT_TRUE(dev.calls.empty());
  EXPECT_EQ(reg->segs.find("P"), reg->segs.end());
}

// KvStagingPool double-pins BOTH its buffers: two engine registrations and two
// NOC maps, each page-aligned. (FakeRegistry keeps only the last base per
// engine name, so assert via the device map, which sees every buffer.)
TEST(DoublePinnedBuffer, StagingPoolDoublePinsBothBuffers) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<FakeTransferEngine>(reg, "P");
  RecordingDeviceMap dev;

  KvStagingPool pool(engine, /*bufferBytes=*/64 * 1024, dev.fn());
  ASSERT_TRUE(pool.registered());
  EXPECT_EQ(pool.bufferBytes(), 64u * 1024);
  ASSERT_NE(pool.buffer(0), nullptr);
  ASSERT_NE(pool.buffer(1), nullptr);

  // One NOC map per buffer (kBuffers == 2), each at the buffer's base, aligned.
  ASSERT_EQ(dev.calls.size(),
            static_cast<std::size_t>(KvStagingPool::kBuffers));
  for (const auto& [va, bytes] : dev.calls) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(va) % DoublePinnedBuffer::kAlign, 0u);
    EXPECT_EQ(bytes, 64u * 1024);
  }
}

}  // namespace
}  // namespace tt::transport
