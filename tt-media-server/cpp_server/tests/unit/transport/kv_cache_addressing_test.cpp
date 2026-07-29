// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "transport/kv_cache_layout.hpp"
#include "transport/kv_cache_mirror.hpp"
#include "transport/multi_device_umd.hpp"
#include "transport/remote_region.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {
namespace {

// Chunk size used throughout: the KvChunkAddressTable default (18 x 1088 bfp8
// tiles). Hex 0x4C80.
constexpr uint64_t K_CHUNK_BYTES = 19584;

// The decode-side ("dst") table from the design's worked example: slot 5, two
// layers across two devices, two position chunks each.
//   (5,0,0) Dec_A ch0 0x800000   (5,0,1) Dec_A ch0 0x804C80
//   (5,1,0) Dec_B ch3 0x900000   (5,1,1) Dec_B ch3 0x904C80
std::vector<KvChunkLocation> workedExampleDstTable() {
  constexpr LocalDeviceId kDecA = 0;
  constexpr LocalDeviceId kDecB = 1;
  return {
      {kDecA, makeNocAddr(0, 0x800000), K_CHUNK_BYTES},
      {kDecA, makeNocAddr(0, 0x804C80), K_CHUNK_BYTES},
      {kDecB, makeNocAddr(3, 0x900000), K_CHUNK_BYTES},
      {kDecB, makeNocAddr(3, 0x904C80), K_CHUNK_BYTES},
  };
}

// A single (device, channel) of contiguous chunks packs at dev_base, offsets
// are exactly (local - base).
TEST(KvCacheLayout, SingleDeviceSingleChannelContiguous) {
  const LocalDeviceId dev = 0;
  std::vector<KvChunkLocation> chunks = {
      {dev, makeNocAddr(0, 0x1000), K_CHUNK_BYTES},
      {dev, makeNocAddr(0, 0x1000 + K_CHUNK_BYTES), K_CHUNK_BYTES},
  };
  KvCacheLayout layout(chunks);

  EXPECT_EQ(layout.numRegions(), 1u);
  EXPECT_EQ(layout.totalBytes(), 2 * K_CHUNK_BYTES);
  EXPECT_EQ(layout.offsetOf(dev, makeNocAddr(0, 0x1000)), 0u);
  EXPECT_EQ(layout.offsetOf(dev, makeNocAddr(0, 0x1000 + K_CHUNK_BYTES)),
            K_CHUNK_BYTES);
}

// The worked example: the documented offsets {0, 19584, 39168, 58752} and total
// size must come out exactly.
TEST(KvCacheLayout, WorkedExampleOffsets) {
  const auto table = workedExampleDstTable();
  KvCacheLayout layout(table);

  // Two regions: (Dec_A, ch0) then (Dec_B, ch3); each spans 2 chunks.
  EXPECT_EQ(layout.numRegions(), 2u);
  EXPECT_EQ(layout.totalBytes(), 4 * K_CHUNK_BYTES);  // 78336

  EXPECT_EQ(layout.offsetOf(0, makeNocAddr(0, 0x800000)), 0u);
  EXPECT_EQ(layout.offsetOf(0, makeNocAddr(0, 0x804C80)), 19584u);
  EXPECT_EQ(layout.offsetOf(1, makeNocAddr(3, 0x900000)), 39168u);
  EXPECT_EQ(layout.offsetOf(1, makeNocAddr(3, 0x904C80)), 58752u);
}

// Regions are packed in a stable (device, channel) order regardless of the
// order chunks are supplied, so both sides agree.
TEST(KvCacheLayout, DeterministicRegardlessOfInputOrder) {
  auto table = workedExampleDstTable();
  KvCacheLayout a(table);
  std::reverse(table.begin(), table.end());
  KvCacheLayout b(table);

  ASSERT_EQ(a.numRegions(), b.numRegions());
  for (std::size_t i = 0; i < a.regions().size(); ++i) {
    EXPECT_EQ(a.regions()[i].device, b.regions()[i].device);
    EXPECT_EQ(a.regions()[i].channel, b.regions()[i].channel);
    EXPECT_EQ(a.regions()[i].seg_base, b.regions()[i].seg_base);
    EXPECT_EQ(a.regions()[i].size, b.regions()[i].size);
  }
}

// Unknown device/channel and out-of-region addresses return nullopt.
TEST(KvCacheLayout, RejectsUnknownAndOutOfRange) {
  KvCacheLayout layout(workedExampleDstTable());

  EXPECT_FALSE(layout.offsetOf(9, makeNocAddr(0, 0x800000)));  // unknown device
  EXPECT_FALSE(
      layout.offsetOf(0, makeNocAddr(7, 0x800000)));  // unknown channel
  EXPECT_FALSE(layout.offsetOf(0, makeNocAddr(0, 0x700000)));  // below dev_base
  // Past the end of the (Dec_A, ch0) region (covers 0x800000 .. +2*chunk).
  EXPECT_FALSE(
      layout.offsetOf(0, makeNocAddr(0, 0x800000 + 2 * K_CHUNK_BYTES)));
}

// The mirror allocates exactly total_bytes and hands out interior pointers that
// match the layout offsets.
TEST(KvCacheMirror, AllocatesAndPointsIntoBuffer) {
  KvCacheMirror mirror(workedExampleDstTable());

  EXPECT_EQ(mirror.totalBytes(), 4 * K_CHUNK_BYTES);
  ASSERT_NE(mirror.base(), nullptr);

  const uint8_t* p = mirror.chunkPtr(1, makeNocAddr(3, 0x900000));
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p - mirror.base(), 39168);

  EXPECT_EQ(mirror.chunkPtr(9, makeNocAddr(0, 0x800000)), nullptr);  // unknown
}

// THE invariant: the sender's RemoteRegion and the receiver's KvCacheMirror,
// built from the same table, agree on every offset. Parity is structural (same
// KvCacheLayout constructor), this guards against regressions.
TEST(RemoteRegion, OffsetParityWithMirror) {
  const auto table = workedExampleDstTable();
  KvCacheMirror mirror(table);
  RemoteRegion remote(/*segment=*/SegmentHandle{42}, table);

  EXPECT_EQ(remote.segment(), SegmentHandle{42});
  EXPECT_EQ(remote.totalBytes(), mirror.totalBytes());

  for (const auto& chunk : table) {
    const auto mirrorOff = mirror.offsetOf(chunk.device, chunk.noc_addr);
    const auto remoteOff = remote.mirrorOffset(chunk.device, chunk.noc_addr);
    ASSERT_TRUE(mirrorOff.has_value());
    ASSERT_TRUE(remoteOff.has_value());
    EXPECT_EQ(*mirrorOff, *remoteOff);
  }
}

// MultiDeviceUmd dispatch bookkeeping is exercised in every build; the actual
// device I/O reports failure without USE_METAL_CPP_LIB (no device), mirroring
// the existing UmdDeviceAccess tests.
TEST(MultiDeviceUmd, DispatchesByDevice) {
  MultiDeviceUmd umd;
  EXPECT_FALSE(umd.hasDevice(0));

  umd.addDevice(0, std::make_shared<UmdDeviceAccess>(/*device_id=*/0));
  umd.addDevice(1, std::make_shared<UmdDeviceAccess>(/*device_id=*/1));
  EXPECT_TRUE(umd.hasDevice(0));
  EXPECT_TRUE(umd.hasDevice(1));
  EXPECT_EQ(umd.numDevices(), 2u);

  std::vector<uint8_t> buffer(K_CHUNK_BYTES, 0);
  // Unknown device: rejected regardless of build.
  EXPECT_FALSE(umd.read(7, makeNocAddr(0, 0), buffer.size(), buffer.data()));
  EXPECT_FALSE(umd.write(7, makeNocAddr(0, 0), buffer.data(), buffer.size()));

#ifndef USE_METAL_CPP_LIB
  // Known device but no real UMD backend in the build -> reports failure.
  EXPECT_FALSE(umd.read(0, makeNocAddr(0, 0), buffer.size(), buffer.data()));
  EXPECT_FALSE(umd.write(0, makeNocAddr(0, 0), buffer.data(), buffer.size()));
#endif
}

}  // namespace
}  // namespace tt::transport
