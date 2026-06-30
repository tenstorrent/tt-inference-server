// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"
#include "transport/transfer_types.hpp"
#include "transport_test_fakes.hpp"

namespace tt::transport {
namespace {

using test::asymmetricReq;
using test::buildTable;
using test::FakeDeviceIo;
using test::FakeRegistry;
using test::FakeTransferEngine;
using test::K_CHUNK;
using test::makeContent;
using test::symmetricReq;
using test::wholeSlot5;

// Decode device whose writes fail while `fail` is set — drives a retryable
// drain.
class ToggleFailDeviceIo : public IDeviceIo {
 public:
  bool fail = true;
  bool read(LocalDeviceId, NocAddr, std::size_t, void*) override {
    return false;
  }
  bool write(LocalDeviceId d, NocAddr n, const void* host,
             std::size_t size) override {
    if (fail) return false;
    const auto* p = static_cast<const uint8_t*>(host);
    store[{d, n}].assign(p, p + size);
    return true;
  }
  std::vector<uint8_t> get(LocalDeviceId d, NocAddr n) const {
    const auto it = store.find({d, n});
    return it == store.end() ? std::vector<uint8_t>{} : it->second;
  }

 private:
  std::map<std::pair<LocalDeviceId, NocAddr>, std::vector<uint8_t>> store;
};

// Full sender -> (simulated one-sided wire) -> receiver -> device round trip.
TEST(MooncakeKvMigration, WholeSlotRoundTripWithFanout) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  // Prefill table: devices on host "P" at addresses 0x1000 / 0x2000.
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  // Decode table: DIFFERENT devices/addresses on host "D" (layer1 on channel
  // 1).
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  // Populate prefill device DRAM with the known content for every replica.
  FakeDeviceIo prefillDev;
  for (uint32_t p = 0; p < 128; p += 32) {
    prefillDev.put(encodeDevice({1, 0}),
                   makeNocAddr(0, 0x1000 + (p / 32) * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 1}),
                   makeNocAddr(0, 0x1000 + (p / 32) * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 2}),
                   makeNocAddr(0, 0x2000 + (p / 32) * K_CHUNK),
                   makeContent(1, p));
    prefillDev.put(encodeDevice({1, 3}),
                   makeNocAddr(0, 0x2000 + (p / 32) * K_CHUNK),
                   makeContent(1, p));
  }
  FakeDeviceIo decodeDev;  // starts empty; drain fills it

  MooncakeKvReceiver receiver(receiverEngine, decodeDev, decode, "D", "D");
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  const uint64_t uuid = 0xABCD;
  const auto seg = receiver.prepareMirror(wholeSlot5().dstSlice(), uuid);
  ASSERT_TRUE(seg.has_value());
  EXPECT_EQ(*seg, "D");
  EXPECT_EQ(receiver.pendingCount(), 1u);

  ASSERT_TRUE(sender.transferSlot(wholeSlot5(), *seg));
  ASSERT_TRUE(receiver.drain(uuid));
  EXPECT_EQ(receiver.pendingCount(), 0u);

  // Every decode replica device now holds the source content for its
  // (layer,pos).
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    const auto l0 = makeContent(0, p);
    const auto l1 = makeContent(1, p);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
              l0);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 1}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
              l0);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
              l1);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 3}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
              l1);
  }
}

// The mirror is registered exactly once, at construction (not per migration),
// survives a drain, and is unregistered when the receiver is destroyed.
TEST(MooncakeKvMigration, MirrorRegisteredOnceAtInit) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<FakeTransferEngine>(reg, "D");
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  FakeDeviceIo dev;
  {
    MooncakeKvReceiver receiver(engine, dev, decode, "D", "D");
    EXPECT_TRUE(receiver.registered());
    // Segment is live before any BeginMigration.
    ASSERT_NE(reg->segs.find("D"), reg->segs.end());
    uint8_t* const base = reg->segs["D"].first;

    // prepareMirror records a plan but does NOT re-register (base unchanged).
    ASSERT_TRUE(receiver.prepareMirror(wholeSlot5().dstSlice(), 1).has_value());
    EXPECT_EQ(reg->segs["D"].first, base);

    // drain forgets the uuid but keeps the segment registered.
    receiver.drain(1);
    EXPECT_NE(reg->segs.find("D"), reg->segs.end());
    EXPECT_EQ(reg->segs["D"].first, base);
  }
  // Destructor unregisters.
  EXPECT_EQ(reg->segs.find("D"), reg->segs.end());
}

// Two concurrent migrations of DISJOINT chunk ranges share the ONE registered
// mirror: both are prepared before either drains, and each lands at its own
// stable offset. Regression guard for the per-request mirror, which registered
// the segment once per migration so the sender wrote into the wrong buffer.
TEST(MooncakeKvMigration, ConcurrentDisjointMigrationsShareOneMirror) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    prefillDev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 1}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
    prefillDev.put(encodeDevice({1, 3}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
  }
  FakeDeviceIo decodeDev;

  MooncakeKvReceiver receiver(receiverEngine, decodeDev, decode, "D", "D");
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  // Disjoint position ranges of slot 5: lo -> chunks at pos 0,32; hi -> 64,96.
  const MigrationRequest lo = symmetricReq(5, 0, 2, 0, 64);
  const MigrationRequest hi = symmetricReq(5, 0, 2, 64, 128);

  // Prepare BOTH before draining either — two live migrations on one mirror.
  const auto segLo = receiver.prepareMirror(lo.dstSlice(), 0xA);
  const auto segHi = receiver.prepareMirror(hi.dstSlice(), 0xB);
  ASSERT_TRUE(segLo.has_value());
  ASSERT_TRUE(segHi.has_value());
  EXPECT_EQ(receiver.pendingCount(), 2u);

  ASSERT_TRUE(sender.transferSlot(lo, *segLo));
  ASSERT_TRUE(sender.transferSlot(hi, *segHi));
  ASSERT_TRUE(receiver.drain(0xA));
  ASSERT_TRUE(receiver.drain(0xB));
  EXPECT_EQ(receiver.pendingCount(), 0u);

  // Both ranges landed on every decode replica at the right (layer, pos) addr.
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    const auto l0 = makeContent(0, p);
    const auto l1 = makeContent(1, p);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
              l0);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 1}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
              l0);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
              l1);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 3}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
              l1);
  }
}

// A failed drain keeps the migration's plan (mirror bytes persist), so
// re-driving the SAME drain after the device recovers completes it with no
// re-transfer.
TEST(MooncakeKvMigration, DrainFailureKeepsPlanAndRetrySucceeds) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    prefillDev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 1}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
    prefillDev.put(encodeDevice({1, 3}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
  }
  ToggleFailDeviceIo decodeDev;  // device writes fail until the flag is cleared

  MooncakeKvReceiver receiver(receiverEngine, decodeDev, decode, "D", "D");
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  const uint64_t uuid = 0xCAFE;
  const auto seg = receiver.prepareMirror(wholeSlot5().dstSlice(), uuid);
  ASSERT_TRUE(seg.has_value());
  ASSERT_TRUE(
      sender.transferSlot(wholeSlot5(), *seg));  // bytes now in the mirror

  // First drain fails on every device write; the plan is KEPT for retry.
  EXPECT_FALSE(receiver.drain(uuid));
  EXPECT_EQ(receiver.pendingCount(), 1u);

  // Device recovers; re-driving the same drain (no new transferSlot) completes.
  decodeDev.fail = false;
  EXPECT_TRUE(receiver.drain(uuid));
  EXPECT_EQ(receiver.pendingCount(), 0u);

  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
              makeContent(0, p));
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
              makeContent(1, p));
  }
}

// Sender fails cleanly if the receiver never advertised a segment.
TEST(MooncakeKvMigration, SenderFailsWithoutMirror) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  FakeDeviceIo dev;
  MooncakeKvSender sender(senderEngine, dev, prefill, decode, "P", "D");
  EXPECT_FALSE(
      sender.transferSlot(wholeSlot5(), "D"));  // no segment registered
}

// A request with a gap is rejected wholesale: prepareMirror must not migrate
// the found subset and report success, which would leave stale KV at the gap on
// decode. The full-table mirror is still non-empty (so the receiver registers),
// only the requested range has a hole.
TEST(MooncakeKvMigration, PrepareMirrorRejectsPartialRequest) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto decode = std::make_shared<InMemoryKvTable>(test::reducedConfig());
  const FabricNode l0a{2, 0}, l0b{2, 1}, l1a{2, 2}, l1b{2, 3};
  const uint32_t g0 = decode->addDeviceGroup({l0a, l0b});
  const uint32_t g1 = decode->addDeviceGroup({l1a, l1b});
  for (const auto& n : {l0a, l0b, l1a, l1b}) decode->setHost(n, "D");
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    // Leave slot 5, layer 0, position 64 absent — the reviewer's gap.
    if (p != 64) {
      decode->setChunk(5, 0, p,
                       {makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK, g0});
    }
    decode->setChunk(5, 1, p,
                     {makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK, g1});
  }

  FakeDeviceIo dev;
  MooncakeKvReceiver receiver(engine, dev, decode, "D", "D");
  EXPECT_FALSE(
      receiver.prepareMirror(wholeSlot5().dstSlice(), /*uuid=*/77).has_value());
}

// Position shift: read src positions [0,64) and land them at dst positions
// [64,128) of the same slot. Source and destination device addresses differ;
// chunks pair by ordinal, so the bytes seeded at src 0/32 must appear at dst
// 64/96. This is the core asymmetric-addressing guard.
TEST(MooncakeKvMigration, PositionShiftRoundTrip) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  // Seed only the src chunks that will be read: positions 0 and 32.
  FakeDeviceIo prefillDev;
  for (uint32_t p = 0; p < 64; p += 32) {
    const uint32_t idx = p / 32;
    prefillDev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
  }
  FakeDeviceIo decodeDev;

  MooncakeKvReceiver receiver(receiverEngine, decodeDev, decode, "D", "D");
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  const MigrationRequest req = asymmetricReq(/*src_slot=*/5, /*dst_slot=*/5,
                                             /*layers=*/0, 2,
                                             /*src_pos=*/0, 64,
                                             /*dst_pos=*/64, 128);
  const uint64_t uuid = 0x5417;
  const auto seg = receiver.prepareMirror(req.dstSlice(), uuid);
  ASSERT_TRUE(seg.has_value());
  ASSERT_TRUE(sender.transferSlot(req, *seg));
  ASSERT_TRUE(receiver.drain(uuid));

  // dst chunk ordinal k holds src chunk ordinal k: dst pos 64<-src 0,
  // 96<-src 32.
  struct Pair {
    uint32_t dstPos;
    uint32_t srcPos;
  };
  for (const Pair pr : {Pair{64, 0}, Pair{96, 32}}) {
    const uint32_t idx = pr.dstPos / 32;
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
              makeContent(0, pr.srcPos))
        << "layer0 dst pos " << pr.dstPos;
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
              makeContent(1, pr.srcPos))
        << "layer1 dst pos " << pr.dstPos;
  }
  // The un-targeted dst chunks (pos 0,32) stay empty — only the range moved.
  EXPECT_TRUE(
      decodeDev.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000)).empty());
}

// Cross-slot: prefill holds the slot the sender reads (src_slot=3), decode
// holds a different slot (dst_slot=5). Guards that the read uses src_slot and
// the write uses dst_slot — a swap would find no chunks on one side.
TEST(MooncakeKvMigration, CrossSlotMigration) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  // Prefill populates slot 3; decode populates slot 5 (distinct addresses).
  auto prefill = std::make_shared<InMemoryKvTable>(buildTable(
      "P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0, /*slot=*/3));
  auto decode = std::make_shared<InMemoryKvTable>(buildTable(
      "D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1, /*slot=*/5));

  FakeDeviceIo prefillDev;
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    prefillDev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
  }
  FakeDeviceIo decodeDev;

  MooncakeKvReceiver receiver(receiverEngine, decodeDev, decode, "D", "D");
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  const MigrationRequest req =
      asymmetricReq(/*src_slot=*/3, /*dst_slot=*/5, 0, 2, 0, 128, 0, 128);
  const uint64_t uuid = 0x3505;
  const auto seg = receiver.prepareMirror(req.dstSlice(), uuid);
  ASSERT_TRUE(seg.has_value());
  ASSERT_TRUE(sender.transferSlot(req, *seg));
  ASSERT_TRUE(receiver.drain(uuid));

  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
              makeContent(0, p));
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
              makeContent(1, p));
  }
}

// The sender rejects a request whose src and dst position ranges cover a
// different number of chunks — there is no 1:1 ordinal pairing.
TEST(MooncakeKvMigration, SenderRejectsChunkCountMismatch) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  FakeDeviceIo prefillDev, decodeDev;
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, decode, "D", "D");
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  // src [0,64) = 2 chunks, dst [0,32) = 1 chunk: mismatched counts.
  const MigrationRequest bad = asymmetricReq(5, 5, 0, 2, 0, 64, 0, 32);
  const auto seg = receiver.prepareMirror(bad.dstSlice(), 0x9001);
  ASSERT_TRUE(seg.has_value());  // dst slice alone is satisfiable...
  EXPECT_FALSE(sender.transferSlot(bad, *seg));  // ...but the pairing is not.
}

// Draining an unknown migration is a clean failure.
TEST(MooncakeKvMigration, DrainUnknownUuidFails) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<FakeTransferEngine>(reg, "D");
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  FakeDeviceIo dev;
  MooncakeKvReceiver receiver(engine, dev, decode, "D", "D");
  EXPECT_FALSE(receiver.drain(/*uuid=*/123));
}

}  // namespace
}  // namespace tt::transport
