// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_table_adapter.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_cache_layout.hpp"
#include "transport/kv_chunk_address_table_adapter.hpp"
#include "transport/kv_table_view.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {
namespace {

constexpr uint64_t K_CHUNK_BYTES = 19584;

// A reduced config mirroring the real decoder table's shape: device groups of 2
// replicas, chunks striped across 2 channels, two layers split across two
// decode hosts.
//   layer 0 -> host "decode-A", group {(m1,c0),(m1,c1)}
//   layer 1 -> host "decode-B", group {(m2,c0),(m2,c1)}
//   per (slot,layer): positions {0,64}->ch0, {32,96}->ch1, at 0x1000 then
//   +chunk
InMemoryKvTable makeReducedTable() {
  KvTableConfig cfg;
  cfg.num_layers = 2;
  cfg.num_slots = 2;
  cfg.max_sequence_length = 128;  // -> 4 position chunks (0,32,64,96)
  cfg.chunk_n_tokens = 32;
  cfg.chunk_size_bytes = static_cast<uint32_t>(K_CHUNK_BYTES);

  InMemoryKvTable table(cfg);
  const FabricNode a0{1, 0}, a1{1, 1}, b0{2, 0}, b1{2, 1};
  const uint32_t g0 = table.addDeviceGroup({a0, a1});
  const uint32_t g1 = table.addDeviceGroup({b0, b1});
  table.setHost(a0, "decode-A");
  table.setHost(a1, "decode-A");
  table.setHost(b0, "decode-B");
  table.setHost(b1, "decode-B");

  auto place = [&](uint32_t group, uint32_t layer) {
    for (uint32_t slot = 0; slot < cfg.num_slots; ++slot) {
      // ch0 holds the first position block, ch1 the second; dense per channel.
      table.setChunk(slot, layer, 0,
                     {makeNocAddr(0, 0x1000), K_CHUNK_BYTES, group});
      table.setChunk(
          slot, layer, 64,
          {makeNocAddr(0, 0x1000 + K_CHUNK_BYTES), K_CHUNK_BYTES, group});
      table.setChunk(slot, layer, 32,
                     {makeNocAddr(1, 0x1000), K_CHUNK_BYTES, group});
      table.setChunk(
          slot, layer, 96,
          {makeNocAddr(1, 0x1000 + K_CHUNK_BYTES), K_CHUNK_BYTES, group});
    }
  };
  place(g0, /*layer=*/0);
  place(g1, /*layer=*/1);
  return table;
}

// The table walk (buildHostPlan / firstUnresolvedChunk / hostsForRequest)
// operates on a per-side KvSlice; these helpers exercise it directly.
KvSlice wholeSlot(uint32_t slot) {
  return KvSlice{slot, /*layer_begin=*/0, /*layer_end=*/2,
                 /*position_begin=*/0, /*position_end=*/128};
}

// hostsForRequest enumerates the decode cluster a whole slot spans.
TEST(KvTableAdapter, HostsForRequestSpansAllHostsOfTheSlot) {
  const auto table = makeReducedTable();
  const auto hosts = hostsForRequest(table, wholeSlot(0));
  ASSERT_EQ(hosts.size(), 2u);
  EXPECT_EQ(hosts[0], "decode-A");
  EXPECT_EQ(hosts[1], "decode-B");
}

// A host's plan contains only that host's layers, with each chunk fanned out to
// both replicas on the host.
TEST(KvTableAdapter, FiltersByHostAndFansOutReplicas) {
  const auto table = makeReducedTable();
  const auto plan = buildHostPlan(table, "decode-A", wholeSlot(0));

  // Layer 0 only (layer 1 is on decode-B); 4 positions.
  ASSERT_EQ(plan.chunks.size(), 4u);
  for (const auto& chunk : plan.chunks) {
    EXPECT_EQ(chunk.layer, 0u);
    ASSERT_EQ(chunk.targets.size(), 2u);  // fan-out to both replicas
    EXPECT_EQ(chunk.targets[0].device, encodeDevice({1, 0}));
    EXPECT_EQ(chunk.targets[1].device, encodeDevice({1, 1}));
  }
  // 4 chunks x 2 replicas = 8 flattened locations.
  EXPECT_EQ(plan.locations.size(), 8u);
}

// Layer and position ranges narrow the plan.
TEST(KvTableAdapter, LayerAndPositionRangesNarrowThePlan) {
  const auto table = makeReducedTable();

  KvSlice req = wholeSlot(0);
  req.position_end = 64;  // positions 0 and 32 only
  const auto plan = buildHostPlan(table, "decode-A", req);
  ASSERT_EQ(plan.chunks.size(), 2u);
  EXPECT_EQ(plan.chunks[0].position, 0u);
  EXPECT_EQ(plan.chunks[1].position, 32u);

  // Layer range excluding layer 0 -> nothing on decode-A.
  req = wholeSlot(0);
  req.layer_begin = 1;
  EXPECT_TRUE(buildHostPlan(table, "decode-A", req).empty());
}

// The plan's flattened locations feed KvCacheLayout: 2 devices x 2 channels = 4
// regions, each spanning 2 dense chunks.
TEST(KvTableAdapter, LocationsFeedKvCacheLayout) {
  const auto table = makeReducedTable();
  const auto plan = buildHostPlan(table, "decode-A", wholeSlot(0));

  KvCacheLayout layout(plan.locations);
  EXPECT_EQ(layout.numRegions(),
            4u);  // (dev0,ch0)(dev0,ch1)(dev1,ch0)(dev1,ch1)
  EXPECT_EQ(layout.totalBytes(), 4 * 2 * K_CHUNK_BYTES);

  // First chunk of dev (1,0) ch0 sits at its region base; the +chunk address is
  // one chunk in.
  const LocalDeviceId dev = encodeDevice({1, 0});
  const auto base = layout.offsetOf(dev, makeNocAddr(0, 0x1000));
  const auto next =
      layout.offsetOf(dev, makeNocAddr(0, 0x1000 + K_CHUNK_BYTES));
  ASSERT_TRUE(base.has_value());
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(*next - *base, K_CHUNK_BYTES);
}

// Absent chunks are skipped (a sparser table than the request range).
TEST(KvTableAdapter, SkipsAbsentChunks) {
  KvTableConfig cfg;
  cfg.num_layers = 1;
  cfg.num_slots = 1;
  cfg.max_sequence_length = 128;
  cfg.chunk_n_tokens = 32;
  cfg.chunk_size_bytes = static_cast<uint32_t>(K_CHUNK_BYTES);
  InMemoryKvTable table(cfg);
  const uint32_t g = table.addDeviceGroup({{1, 0}});
  table.setHost({1, 0}, "decode-A");
  // Only position 32 present; 0/64/96 absent.
  table.setChunk(0, 0, 32, {makeNocAddr(0, 0x1000), K_CHUNK_BYTES, g});

  const auto plan = buildHostPlan(table, "decode-A", KvSlice{0, 0, 1, 0, 128});
  ASSERT_EQ(plan.chunks.size(), 1u);
  EXPECT_EQ(plan.chunks[0].position, 32u);
  EXPECT_EQ(plan.locations.size(), 1u);
}

// firstUnresolvedChunk passes when every requested chunk resolves, regardless
// of which host holds it (it checks table presence, not placement).
TEST(KvTableAdapter, FirstUnresolvedChunkPassesWhenFullyResolved) {
  const auto table = makeReducedTable();
  EXPECT_FALSE(firstUnresolvedChunk(table, wholeSlot(0)).has_value());
}

// A request whose range includes a chunk the table does not hold is reported,
// pinpointing the first gap — even though buildHostPlan would silently drop it
// and return a non-empty plan.
TEST(KvTableAdapter, FirstUnresolvedChunkReportsAbsentChunk) {
  KvTableConfig cfg;
  cfg.num_layers = 1;
  cfg.num_slots = 1;
  cfg.max_sequence_length = 128;
  cfg.chunk_n_tokens = 32;
  cfg.chunk_size_bytes = static_cast<uint32_t>(K_CHUNK_BYTES);
  InMemoryKvTable table(cfg);
  const uint32_t g = table.addDeviceGroup({{1, 0}});
  table.setHost({1, 0}, "decode-A");
  // Positions 0, 32, 96 present; 64 absent (the reviewer's example).
  table.setChunk(0, 0, 0, {makeNocAddr(0, 0x1000), K_CHUNK_BYTES, g});
  table.setChunk(0, 0, 32, {makeNocAddr(1, 0x1000), K_CHUNK_BYTES, g});
  table.setChunk(0, 0, 96, {makeNocAddr(1, 0x2000), K_CHUNK_BYTES, g});

  const KvSlice req{0, 0, 1, 0, 128};
  // buildHostPlan happily returns the found subset, masking the gap.
  ASSERT_EQ(buildHostPlan(table, "decode-A", req).chunks.size(), 3u);

  const auto missing = firstUnresolvedChunk(table, req);
  ASSERT_TRUE(missing.has_value());
  EXPECT_EQ(missing->layer, 0u);
  EXPECT_EQ(missing->position, 64u);
}

// Out-of-range layer/position requests resolve to absent chunks, so they are
// rejected too.
TEST(KvTableAdapter, FirstUnresolvedChunkReportsOutOfRange) {
  const auto table = makeReducedTable();

  // position_end beyond max_sequence_length: the table holds 0..96, so the
  // first out-of-range chunk (128) is reported.
  KvSlice req = wholeSlot(0);
  req.position_end = 160;
  auto missing = firstUnresolvedChunk(table, req);
  ASSERT_TRUE(missing.has_value());
  EXPECT_EQ(missing->position, 128u);

  // layer_end beyond num_layers: layer 2 does not exist.
  req = wholeSlot(0);
  req.layer_end = 3;
  missing = firstUnresolvedChunk(table, req);
  ASSERT_TRUE(missing.has_value());
  EXPECT_EQ(missing->layer, 2u);
}

// MigrationRequest projects to per-side slices: layers are shared, slot and
// position range come from the matching side. Everything downstream keys off
// these, so guard the projection directly.
TEST(KvTableAdapter, MigrationRequestSlicesProject) {
  const MigrationRequest r{/*src_slot=*/3,
                           /*dst_slot=*/5,
                           /*layer_begin=*/1,
                           /*layer_end=*/4,
                           /*src_position_begin=*/0,
                           /*src_position_end=*/64,
                           /*dst_position_begin=*/128,
                           /*dst_position_end=*/192};
  const KvSlice s = r.srcSlice();
  EXPECT_EQ(s.slot, 3u);
  EXPECT_EQ(s.layer_begin, 1u);
  EXPECT_EQ(s.layer_end, 4u);
  EXPECT_EQ(s.position_begin, 0u);
  EXPECT_EQ(s.position_end, 64u);
  const KvSlice d = r.dstSlice();
  EXPECT_EQ(d.slot, 5u);
  EXPECT_EQ(d.layer_begin, 1u);  // shared with src
  EXPECT_EQ(d.layer_end, 4u);
  EXPECT_EQ(d.position_begin, 128u);
  EXPECT_EQ(d.position_end, 192u);
}

// A shifted dst slice addresses the shifted chunk range: dst positions [64,128)
// resolve to the table's pos-64 and pos-96 chunks (different channels/addresses
// from the [0,64) chunks the src side would read). This is the addressing half
// of a position shift; the byte-level pairing is covered in the migration test.
TEST(KvTableAdapter, ShiftedDstSliceAddressesShiftedChunks) {
  const auto table = makeReducedTable();
  KvSlice dst = wholeSlot(0);
  dst.position_begin = 64;  // -> positions 64, 96
  const auto plan = buildHostPlan(table, "decode-A", dst);

  ASSERT_EQ(plan.chunks.size(), 2u);
  EXPECT_EQ(plan.chunks[0].position, 64u);
  EXPECT_EQ(plan.chunks[1].position, 96u);
  // pos 64 lives on ch0 at 0x1000+chunk; pos 96 on ch1 at 0x1000+chunk.
  EXPECT_EQ(plan.chunks[0].targets[0].noc_addr,
            makeNocAddr(0, 0x1000 + K_CHUNK_BYTES));
  EXPECT_EQ(plan.chunks[1].targets[0].noc_addr,
            makeNocAddr(1, 0x1000 + K_CHUNK_BYTES));
}

// The real-table adapter implements the same IKvTable interface buildHostPlan
// consumes. Without the table build guard, its factories no-op so callers fall
// back to InMemoryKvTable; with the guard, the integration tests exercise the
// real protobuf path instead.
#ifndef TT_TRANSPORT_WITH_KV_TABLE
TEST(KvChunkAddressTableAdapter, FallbackWhenGuardOff) {
  EXPECT_FALSE(KvChunkAddressTableAdapter::available());
  EXPECT_EQ(KvChunkAddressTableAdapter::fromProtobuf(""), nullptr);
  EXPECT_EQ(KvChunkAddressTableAdapter::fromProtobufFile("/nonexistent.pb"),
            nullptr);
}
#endif

// The adapter is an IKvTable, so buildHostPlan accepts it interchangeably with
// InMemoryKvTable — checked at compile time (no instance needed when guarded
// off).
static_assert(std::is_base_of_v<IKvTable, KvChunkAddressTableAdapter>,
              "KvChunkAddressTableAdapter must implement IKvTable");

}  // namespace
}  // namespace tt::transport
