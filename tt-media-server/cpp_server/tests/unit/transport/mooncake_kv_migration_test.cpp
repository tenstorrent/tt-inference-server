// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "transport/double_pinned_buffer.hpp"
#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_bounce_buffer.hpp"
#include "transport/kv_control_message.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"
#include "transport/transfer_types.hpp"
#include "transport_test_fakes.hpp"

namespace tt::transport {
namespace {

using test::asymmetricReq;
using test::AsyncSpanDeviceIo;
using test::buildTable;
using test::FakeDeviceIo;
using test::FakeRegistry;
using test::FakeTransferEngine;
using test::K_CHUNK;
using test::makeContent;
using test::reducedConfig;
using test::SpanDeviceIo;
using test::symmetricReq;
using test::wholeSlot5;

// Seed the prefill device with the known content for every replica of a
// whole-slot migration (both layers, both replicas, 4 positions).
void seedWholeSlot(FakeDeviceIo& dev) {
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    dev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
            makeContent(0, p));
    dev.put(encodeDevice({1, 1}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
            makeContent(0, p));
    dev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
            makeContent(1, p));
    dev.put(encodeDevice({1, 3}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
            makeContent(1, p));
  }
}

// Assert every decode replica holds the source content for its (layer, pos).
void expectWholeSlotLanded(const SpanDeviceIo& decodeDev) {
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    const auto l0 = makeContent(0, p);
    const auto l1 = makeContent(1, p);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK),
              l0);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 1}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK),
              l0);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK),
              l1);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 3}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK),
              l1);
  }
}

// A sink that drains straight into the bounce receiver (no channel): the
// sender's WRITE has already landed the bytes in the receiver's bounce buffer
// (shared via the FakeRegistry), so drainWindow copies them to the decode
// devices. Counts the windows so a test can assert the migration actually
// streamed.
struct DrainingSink {
  MooncakeKvReceiver* receiver;
  int windows = 0;
  bool ok = true;

  bool operator()(uint64_t,
                  const std::vector<BounceSectionDescriptor>& window) {
    ++windows;
    const bool drained = receiver->drainWindow(window);
    ok = ok && drained;
    return drained;
  }
};

// Whole-slot round trip through a bounce buffer big enough to hold it in one
// window. section_size 256 lets each layer's 4 contiguous 64B chunks merge into
// one slot.
TEST(MooncakeKvMigration, WholeSlotSingleWindow) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  SpanDeviceIo decodeDev;

  const BounceGeometry geo{/*section_count=*/4, /*section_size=*/256};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  ASSERT_TRUE(receiver.registered());
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  DrainingSink sink{&receiver};
  ASSERT_TRUE(
      sender.transferSlot(0xABCD, wholeSlot5(), "D", geo, std::ref(sink)));
  EXPECT_TRUE(sink.ok);
  // Two merged segments (one per layer) fit in a single 4-slot window.
  EXPECT_EQ(sink.windows, 1);
  expectWholeSlotLanded(decodeDev);
}

// The same migration through a 1-slot bounce buffer must stream over MANY
// windows, each drained and its credit returned before the next — the
// multi-window / backpressure path.
TEST(MooncakeKvMigration, WholeSlotManyWindows) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  SpanDeviceIo decodeDev;

  // section_size 64 == one chunk: no merging, one slot per window -> 8 windows
  // (2 layers x 4 positions).
  const BounceGeometry geo{/*section_count=*/1, /*section_size=*/K_CHUNK};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  DrainingSink sink{&receiver};
  ASSERT_TRUE(
      sender.transferSlot(0xABCE, wholeSlot5(), "D", geo, std::ref(sink)));
  EXPECT_EQ(sink.windows, 8);
  expectWholeSlotLanded(decodeDev);
}

// Position shift over the bounce buffer: src [0,64) lands at dst [64,128),
// paired by ordinal. The core asymmetric-addressing guard for the bounce path.
TEST(MooncakeKvMigration, PositionShift) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  for (uint32_t p = 0; p < 64; p += 32) {
    const uint32_t idx = p / 32;
    prefillDev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
  }
  SpanDeviceIo decodeDev;

  const BounceGeometry geo{4, 256};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  const MigrationRequest req = asymmetricReq(5, 5, 0, 2, 0, 64, 64, 128);
  DrainingSink sink{&receiver};
  ASSERT_TRUE(sender.transferSlot(0x5417, req, "D", geo, std::ref(sink)));

  for (const auto pr : {std::pair<uint32_t, uint32_t>{64, 0}, {96, 32}}) {
    const uint32_t idx = pr.first / 32;
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK),
              makeContent(0, pr.second));
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK),
              makeContent(1, pr.second));
  }
  // The un-targeted dst chunk (pos 0) stays empty — only the range moved.
  EXPECT_TRUE(
      decodeDev.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000), K_CHUNK)
          .empty());
}

// A window whose drain fails propagates false up through the sink so the
// migration aborts (no false success).
TEST(MooncakeKvMigration, DrainFailureAbortsMigration) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);

  // A device whose writes always fail drives drainWindow -> false.
  class FailWriteDeviceIo : public FakeDeviceIo {
   public:
    bool write(LocalDeviceId, NocAddr, const void*, std::size_t) override {
      return false;
    }
  } decodeDev;

  const BounceGeometry geo{4, 256};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  DrainingSink sink{&receiver};
  EXPECT_FALSE(
      sender.transferSlot(0xDEAD, wholeSlot5(), "D", geo, std::ref(sink)));
}

// A transfer engine whose WRITE batch does NOT land until waitBatch(), so a
// window's WRITE is genuinely "in flight" between submitBatch and waitBatch —
// letting a test observe the sender staging the NEXT window during that gap
// (the read∥network overlap). Bytes still land (on waitBatch) before the
// receiver drains, so byte-verify holds.
class DeferredFakeEngine : public FakeTransferEngine {
 public:
  using FakeTransferEngine::FakeTransferEngine;
  std::atomic<bool> writeInFlight{false};

  TransferHandle submitBatch(const std::vector<TransferRequest>& rs) override {
    pending = rs;  // remember; apply on waitBatch, not now
    writeInFlight.store(true);
    return {static_cast<uint64_t>(rs.size()), true};
  }
  TransferStatus waitBatch(TransferHandle h) override {
    if (!h.valid) return {TransferState::FAILED, 0};
    for (const TransferRequest& r : pending) {
      if (FakeTransferEngine::submitAndWait(r).state !=
          TransferState::COMPLETED)
        return {TransferState::FAILED, 0};
    }
    pending.clear();
    writeInFlight.store(false);
    return {TransferState::COMPLETED, 0};
  }

 private:
  std::vector<TransferRequest> pending;
};

// A prefill device that records how many reads happened while a WRITE batch was
// in flight on `eng` — evidence the sender pipelined device reads under the
// previous window's network transfer.
class OverlapDevice : public FakeDeviceIo {
 public:
  DeferredFakeEngine* eng = nullptr;
  std::atomic<int> readsDuringWrite{0};
  bool read(LocalDeviceId d, NocAddr n, std::size_t size, void* host) override {
    if (eng != nullptr && eng->writeInFlight.load()) ++readsDuringWrite;
    return FakeDeviceIo::read(d, n, size, host);
  }
};

// The sender stages the next window's device reads WHILE the current
// window's WRITE batch is in flight. With a deferred engine (WRITE lands only
// on waitBatch) and an 8-window migration, at least one read must occur while a
// WRITE is in flight — and the bytes still land correctly.
TEST(MooncakeKvMigration, PipelineOverlapsReadsWithInFlightWrites) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<DeferredFakeEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  OverlapDevice prefillDev;
  prefillDev.eng = senderEngine.get();
  seedWholeSlot(prefillDev);
  SpanDeviceIo decodeDev;

  // 1 slot / 64 B -> 8 windows, so windows 1..7 stage while the prior WRITE
  // is in flight.
  const BounceGeometry geo{/*section_count=*/1, /*section_size=*/K_CHUNK};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  DrainingSink sink{&receiver};
  ASSERT_TRUE(
      sender.transferSlot(0xF00D, wholeSlot5(), "D", geo, std::ref(sink)));
  EXPECT_GT(prefillDev.readsDuringWrite.load(), 0)
      << "expected reads to overlap in-flight WRITEs (pipeline)";
  expectWholeSlotLanded(decodeDev);
}

// The bounce receiver NOC-maps its bounce buffer (the double-pin device-map
// hook) once at construction, over the whole bounce buffer — the SAME buffer
// the engine registered. On HW this is DriscDeviceIo::registerHostRegion; here
// a recorder.
TEST(MooncakeKvMigration, ReceiverDoublePinsBounceBuffer) {
  auto reg = std::make_shared<FakeRegistry>();
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");
  SpanDeviceIo decodeDev;
  const BounceGeometry geo{4, 256};

  std::vector<std::pair<void*, std::size_t>> maps;
  DeviceMapFn dm = [&](void* va, std::size_t n) { maps.emplace_back(va, n); };

  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo, dm);
  ASSERT_TRUE(receiver.registered());
  // Exactly one NOC map, covering the whole bounce buffer, at the registered
  // base.
  ASSERT_EQ(maps.size(), 1u);
  ASSERT_NE(reg->segs.find("D"), reg->segs.end());
  EXPECT_EQ(maps[0].first, reg->segs["D"].first);
  EXPECT_EQ(maps[0].second, geo.capacity());
}

// A chunk larger than a bounce section is rejected cleanly (no partial
// transfer).
TEST(MooncakeKvMigration, ChunkTooBigForSlotFails) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  FakeDeviceIo decodeDev;

  // section_size 32 < the 64-byte chunk -> transferSlot must fail up front.
  const BounceGeometry geo{4, 32};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  DrainingSink sink{&receiver};
  EXPECT_FALSE(
      sender.transferSlot(0xBEEF, wholeSlot5(), "D", geo, std::ref(sink)));
}

// The DRISC async device path: readAsync reports BUSY while a read is in flight
// (a global 1-in-flight cap here), forcing the sender's tryPopCompleted()+retry
// backpressure loop, and the post-window drain loop retires the outstanding
// read. Exercises the machinery every other test skips via the synchronous
// default — and the bytes must still land.
TEST(MooncakeKvMigration, AsyncDeviceBackpressureAndDrain) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  AsyncSpanDeviceIo prefillDev(/*maxInFlight=*/1);
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    prefillDev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
                   makeContent(0, p));
    prefillDev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
                   makeContent(1, p));
  }
  SpanDeviceIo decodeDev;

  // 4x256 B: the whole slot is one window of 2 merged segments read
  // back-to-back, so the 2nd readAsync hits BUSY on the 1-in-flight device.
  const BounceGeometry geo{/*section_count=*/4, /*section_size=*/256};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  DrainingSink sink{&receiver};
  ASSERT_TRUE(
      sender.transferSlot(0xA5A5, wholeSlot5(), "D", geo, std::ref(sink)));
  EXPECT_GT(prefillDev.busyRejections(), 0u)
      << "expected readAsync to report BUSY (backpressure retry path)";
  EXPECT_GT(prefillDev.deferredReads(), 0u)
      << "expected reads to be retired via tryPopCompleted()";
  expectWholeSlotLanded(decodeDev);
}

// drainWindow bounds-checks each sender-supplied descriptor (untrusted — it
// arrives over the network). A section whose [offset, offset+size) escapes the
// bounce buffer is rejected and nothing is written to the device.
TEST(MooncakeKvMigration, DrainWindowRejectsEscapingDescriptor) {
  auto reg = std::make_shared<FakeRegistry>();
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");
  SpanDeviceIo decodeDev;
  const BounceGeometry geo{/*section_count=*/2, /*section_size=*/128};  // 256 B
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  ASSERT_TRUE(receiver.registered());

  const DrainTarget tgt{encodeDevice({2, 0}), makeNocAddr(0, 0x8000)};

  // In-range offset but the size runs off the end (192 + 128 > 256).
  std::vector<BounceSectionDescriptor> overrun{{192, 128, {tgt}}};
  EXPECT_FALSE(receiver.drainWindow(overrun));
  EXPECT_TRUE(decodeDev.get(tgt.device, tgt.noc_addr, 128).empty());

  // A wildly out-of-range offset is also rejected.
  std::vector<BounceSectionDescriptor> farOffset{{1ull << 20, 64, {tgt}}};
  EXPECT_FALSE(receiver.drainWindow(farOffset));
  EXPECT_TRUE(decodeDev.get(tgt.device, tgt.noc_addr, 64).empty());
}

// An engine that claims a DIFFERENT buffer is buffers[0] than the one just
// registered, tripping the receiver's "bounce buffer must be buffers[0]"
// invariant (remote WRITEs resolve against buffers[0]).
class WrongFirstBufferEngine : public FakeTransferEngine {
 public:
  using FakeTransferEngine::FakeTransferEngine;
  std::size_t registeredLocalBufferCount() const override { return 1; }
  void* firstRegisteredLocalBuffer() const override { return &sentinel; }

 private:
  mutable int sentinel = 0;
};

// If the bounce buffer is not buffers[0] after registration, the receiver
// aborts (unregisters, reports not-registered) rather than silently accepting
// a segment remote WRITEs would resolve against the wrong buffer.
TEST(MooncakeKvMigration, ReceiverRejectsWhenBounceIsNotFirstBuffer) {
  auto reg = std::make_shared<FakeRegistry>();
  auto engine = std::make_shared<WrongFirstBufferEngine>(reg, "D");
  SpanDeviceIo decodeDev;
  const BounceGeometry geo{4, 256};
  MooncakeKvReceiver receiver(engine, decodeDev, "D", geo);
  EXPECT_FALSE(receiver.registered());
  // The guard unregistered the bounce buffer on abort.
  EXPECT_EQ(reg->segs.find("D"), reg->segs.end());
}

// src range covers 2 chunks, dst range 1 -> unequal chunk counts, rejected up
// front (no transfer, no false success).
TEST(MooncakeKvMigration, ChunkCountMismatchFails) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  SpanDeviceIo decodeDev;

  const BounceGeometry geo{4, 256};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  // src [0,64) = 2 chunks; dst [0,32) = 1 chunk.
  const MigrationRequest req = asymmetricReq(5, 5, 0, 2, 0, 64, 0, 32);
  DrainingSink sink{&receiver};
  EXPECT_FALSE(sender.transferSlot(0xC0DE, req, "D", geo, std::ref(sink)));
}

// A dst chunk whose size differs from its paired src chunk is rejected during
// planning (here the decode table places half-size chunks).
TEST(MooncakeKvMigration, SrcDstChunkSizeMismatchFails) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));

  // Decode table identical in layout but with HALF-size chunks.
  auto decode = std::make_shared<InMemoryKvTable>(reducedConfig());
  const uint32_t g0 = decode->addDeviceGroup({{2, 0}, {2, 1}});
  const uint32_t g1 = decode->addDeviceGroup({{2, 2}, {2, 3}});
  for (const auto& n : {FabricNode{2, 0}, FabricNode{2, 1}, FabricNode{2, 2},
                        FabricNode{2, 3}}) {
    decode->setHost(n, "D");
  }
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    decode->setChunk(
        5, 0, p,
        ChunkLoc{makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK / 2, g0});
    decode->setChunk(
        5, 1, p,
        ChunkLoc{makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK / 2, g1});
  }

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  SpanDeviceIo decodeDev;

  const BounceGeometry geo{4, 256};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  DrainingSink sink{&receiver};
  EXPECT_FALSE(
      sender.transferSlot(0x512E, wholeSlot5(), "D", geo, std::ref(sink)));
}

}  // namespace
}  // namespace tt::transport
