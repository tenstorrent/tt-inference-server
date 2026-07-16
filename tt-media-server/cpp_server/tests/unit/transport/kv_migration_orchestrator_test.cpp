// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_orchestrator.hpp"

#include <gtest/gtest.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "services/remote_kv_manager.hpp"
#include "sockets/i_socket_transport.hpp"
#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_migration_multi_host_sender.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"
#include "transport/mooncake_migration_executor.hpp"
#include "transport/transfer_types.hpp"
#include "transport_test_fakes.hpp"

namespace tt::transport {
namespace {

using test::asymmetricReq;
using test::BlockingFakeTransport;
using test::buildTable;
using test::buildTableSplitHosts;
using test::closePipe;
using test::FakeDeviceIo;
using test::FakeRegistry;
using test::FakeTransferEngine;
using test::K_CHUNK;
using test::makeContent;
using test::Pipe;
using test::symmetricReq;
using test::wholeSlot5;

// A decode device whose writes always fail, to drive the drain-failure path.
class FailingWriteDeviceIo : public FakeDeviceIo {
 public:
  bool write(LocalDeviceId, NocAddr, const void*, std::size_t) override {
    return false;
  }
};

// One decode host for the multiHost tests: its own Mooncake engine/segment,
// control channel pair, receiver, and a thread running the receiver
// orchestrator. The device is owned by the test (so the failure-injection
// variant can be used) and must outlive the node. shutdown() (also run by the
// dtor) closes the channel and joins the thread.
struct DecodeNode {
  std::shared_ptr<FakeTransferEngine> engine;
  std::shared_ptr<Pipe> ab{std::make_shared<Pipe>()};  // sender -> receiver
  std::shared_ptr<Pipe> ba{std::make_shared<Pipe>()};  // receiver -> sender
  std::shared_ptr<BlockingFakeTransport> senderTp;
  std::shared_ptr<BlockingFakeTransport> receiverTp;
  std::shared_ptr<KvControlChannel> senderCh;
  std::unique_ptr<KvControlChannel> receiverCh;
  std::unique_ptr<MooncakeKvReceiver> receiver;
  std::unique_ptr<KvMigrationReceiver> orch;
  std::thread thread;

  DecodeNode(const std::shared_ptr<FakeRegistry>& reg,
             const std::shared_ptr<const IKvTable>& table,
             const std::string& host, const std::string& seg, IDeviceIo& dev) {
    engine = std::make_shared<FakeTransferEngine>(reg, seg);
    senderTp = std::make_shared<BlockingFakeTransport>(/*in=*/ba, /*out=*/ab);
    receiverTp = std::make_shared<BlockingFakeTransport>(/*in=*/ab, /*out=*/ba);
    senderCh = std::make_shared<KvControlChannel>(senderTp);
    receiverCh = std::make_unique<KvControlChannel>(receiverTp);
    receiver =
        std::make_unique<MooncakeKvReceiver>(engine, dev, table, host, seg);
    orch = std::make_unique<KvMigrationReceiver>(*receiverCh, *receiver);
    thread = std::thread([this] { orch->run(); });
  }
  void shutdown() {
    closePipe(ab);
    if (thread.joinable()) thread.join();
  }
  ~DecodeNode() { shutdown(); }
};

// The orchestrators drive a complete migration over the control channel: the
// receiver runs on its own thread reacting to BeginMigration / DoneMarker while
// the sender drives the sequence, and the bytes land on the decode devices.
TEST(KvMigrationOrchestrator, DrivesWholeMigrationOverChannel) {
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

  auto ab = std::make_shared<Pipe>();  // sender -> receiver
  auto ba = std::make_shared<Pipe>();  // receiver -> sender
  auto senderTp =
      std::make_shared<BlockingFakeTransport>(/*in=*/ba, /*out=*/ab);
  auto receiverTp =
      std::make_shared<BlockingFakeTransport>(/*in=*/ab, /*out=*/ba);
  KvControlChannel senderCh(senderTp);
  KvControlChannel receiverCh(receiverTp);

  KvMigrationSender senderOrch(senderCh, sender);
  KvMigrationReceiver receiverOrch(receiverCh, receiver);

  std::thread receiverThread([&] { receiverOrch.run(); });

  const uint64_t uuid = 0x5151;
  EXPECT_TRUE(senderOrch.migrate(uuid, wholeSlot5()));

  closePipe(ab);  // unblock the receiver's next receive() so run() returns
  receiverThread.join();

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

// A position shift driven over the channel: BeginMigration carries the dst
// coordinates ([64,128)), the receiver prepares/drains its dst chunks, and the
// bytes the sender read from src [0,64) land at dst 64/96. Guards the wire path
// (dst slice over the channel) end to end, not just the in-process sender.
TEST(KvMigrationOrchestrator, DrivesPositionShiftOverChannel) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;  // seed only the src chunks read: positions 0, 32
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

  auto ab = std::make_shared<Pipe>();
  auto ba = std::make_shared<Pipe>();
  auto senderTp = std::make_shared<BlockingFakeTransport>(ba, ab);
  auto receiverTp = std::make_shared<BlockingFakeTransport>(ab, ba);
  KvControlChannel senderCh(senderTp);
  KvControlChannel receiverCh(receiverTp);
  KvMigrationSender senderOrch(senderCh, sender);
  KvMigrationReceiver receiverOrch(receiverCh, receiver);

  std::thread receiverThread([&] { receiverOrch.run(); });

  const MigrationRequest shift = asymmetricReq(5, 5, 0, 2, 0, 64, 64, 128);
  EXPECT_TRUE(senderOrch.migrate(0x5417, shift));

  closePipe(ab);
  receiverThread.join();

  // dst pos 64 <- src 0, dst pos 96 <- src 32 (ordinal pairing over the wire).
  EXPECT_EQ(
      decodeDev.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000 + 2 * K_CHUNK)),
      makeContent(0, 0));
  EXPECT_EQ(
      decodeDev.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000 + 3 * K_CHUNK)),
      makeContent(0, 32));
  EXPECT_EQ(
      decodeDev.get(encodeDevice({2, 2}), makeNocAddr(1, 0x9000 + 2 * K_CHUNK)),
      makeContent(1, 0));
}

// Init-time table exchange swaps the two sides' table blobs.
TEST(KvMigrationOrchestrator, ExchangesTableBlobs) {
  auto reg = std::make_shared<FakeRegistry>();
  auto se = std::make_shared<FakeTransferEngine>(reg, "P");
  auto re = std::make_shared<FakeTransferEngine>(reg, "D");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  FakeDeviceIo pdev, ddev;
  MooncakeKvSender sender(se, pdev, prefill, decode, "P", "D");
  MooncakeKvReceiver receiver(re, ddev, decode, "D", "D");

  auto ab = std::make_shared<Pipe>();
  auto ba = std::make_shared<Pipe>();
  auto stp = std::make_shared<BlockingFakeTransport>(ba, ab);
  auto rtp = std::make_shared<BlockingFakeTransport>(ab, ba);
  KvControlChannel sch(stp), rch(rtp);
  KvMigrationSender so(sch, sender);
  KvMigrationReceiver ro(rch, receiver);

  const std::vector<uint8_t> senderBlob{0xAA, 0xBB};
  const std::vector<uint8_t> receiverBlob{1, 2, 3, 4};

  std::optional<std::vector<uint8_t>> gotOnReceiver;
  std::thread t([&] { gotOnReceiver = ro.exchangeTables(receiverBlob); });
  const auto gotOnSender = so.exchangeTables(senderBlob);
  t.join();

  ASSERT_TRUE(gotOnSender.has_value());
  ASSERT_TRUE(gotOnReceiver.has_value());
  EXPECT_EQ(*gotOnSender, receiverBlob);
  EXPECT_EQ(*gotOnReceiver, senderBlob);
  EXPECT_EQ(ro.peerTableBlob(), senderBlob);
}

// A prepareMirror failure (no local chunks for the request) is reported to the
// sender via MirrorReady.ok = false, so migrate() fails instead of writing into
// a mirror the receiver never prepared.
TEST(KvMigrationOrchestrator, ReportsMirrorPrepareFailure) {
  auto reg = std::make_shared<FakeRegistry>();
  auto se = std::make_shared<FakeTransferEngine>(reg, "P");
  auto re = std::make_shared<FakeTransferEngine>(reg, "D");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  FakeDeviceIo pdev, ddev;
  MooncakeKvReceiver receiver(re, ddev, decode, "D", "D");
  MooncakeKvSender sender(se, pdev, prefill, decode, "P", "D");

  auto ab = std::make_shared<Pipe>();
  auto ba = std::make_shared<Pipe>();
  auto stp = std::make_shared<BlockingFakeTransport>(ba, ab);
  auto rtp = std::make_shared<BlockingFakeTransport>(ab, ba);
  KvControlChannel sch(stp), rch(rtp);
  KvMigrationSender so(sch, sender);
  KvMigrationReceiver ro(rch, receiver);

  std::thread receiverThread([&] { ro.run(); });

  // Slot 6 has no chunks in either table -> prepareMirror yields an empty plan.
  const MigrationRequest noChunks = symmetricReq(6, 0, 2, 0, 128);
  EXPECT_FALSE(so.migrate(0x6262, noChunks));

  closePipe(ab);  // unblock the receiver's next receive() so run() returns
  receiverThread.join();
}

// A failed device drain on the receiver is reported via Ack.ok = false, so the
// sender's migrate() fails rather than treating a corrupt cache as a success.
TEST(KvMigrationOrchestrator, ReportsDrainFailure) {
  auto reg = std::make_shared<FakeRegistry>();
  auto se = std::make_shared<FakeTransferEngine>(reg, "P");
  auto re = std::make_shared<FakeTransferEngine>(reg, "D");
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
  FailingWriteDeviceIo decodeDev;  // every drain write fails

  MooncakeKvReceiver receiver(re, decodeDev, decode, "D", "D");
  MooncakeKvSender sender(se, prefillDev, prefill, decode, "P", "D");

  auto ab = std::make_shared<Pipe>();
  auto ba = std::make_shared<Pipe>();
  auto stp = std::make_shared<BlockingFakeTransport>(ba, ab);
  auto rtp = std::make_shared<BlockingFakeTransport>(ab, ba);
  KvControlChannel sch(stp), rch(rtp);
  KvMigrationSender so(sch, sender);
  KvMigrationReceiver ro(rch, receiver);

  std::thread receiverThread([&] { ro.run(); });

  EXPECT_FALSE(so.migrate(0x7373, wholeSlot5()));

  closePipe(ab);  // unblock the receiver's next receive() so run() returns
  receiverThread.join();
}

// --- Multi-host sender (n->m fan-out)
// -------------------------------------------

// Seed the prefill device with the primary-replica content for both layers.
void seedPrefill(FakeDeviceIo& dev) {
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    dev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
            makeContent(0, p));
    dev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
            makeContent(1, p));
  }
}

// Whole-slot migration fans out to TWO decode hosts: layer 0 lands on D0, layer
// 1 on D1, each over its own control channel + segment, byte-verified.
TEST(KvMigrationMultiHost, FansOutToTwoDecodeHosts) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  // layer 0 -> host "D0" (mesh 2), layer 1 -> host "D1" (mesh 3).
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0, devD1;
  DecodeNode d0(reg, decode, "D0", "segD0", devD0);
  DecodeNode d1(reg, decode, "D1", "segD1", devD1);

  KvMigrationMultiHostSender multiHost(
      senderEngine, prefillDev, prefill, decode, "P",
      {{"D0", d0.senderCh}, {"D1", d1.senderCh}});
  EXPECT_EQ(multiHost.hostCount(), 2u);
  EXPECT_TRUE(multiHost.migrate(0x5151, wholeSlot5()));

  d0.shutdown();
  d1.shutdown();

  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    // D0 holds layer 0 on both its replicas; D1 holds layer 1.
    EXPECT_EQ(
        devD0.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
        makeContent(0, p));
    EXPECT_EQ(
        devD0.get(encodeDevice({2, 1}), makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
        makeContent(0, p));
    EXPECT_EQ(
        devD1.get(encodeDevice({3, 0}), makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
        makeContent(1, p));
    EXPECT_EQ(
        devD1.get(encodeDevice({3, 1}), makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
        makeContent(1, p));
  }
}

// The multiHost drives ONLY the hosts the request touches: a layer-0-only
// request lands on D0 and never contacts D1 (its device stays empty).
TEST(KvMigrationMultiHost, DrivesOnlyHostsInRequest) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0, devD1;
  DecodeNode d0(reg, decode, "D0", "segD0", devD0);
  DecodeNode d1(reg, decode, "D1", "segD1", devD1);

  KvMigrationMultiHostSender multiHost(
      senderEngine, prefillDev, prefill, decode, "P",
      {{"D0", d0.senderCh}, {"D1", d1.senderCh}});

  // Layer [0,1) lives only on D0.
  EXPECT_TRUE(multiHost.migrate(0x1, symmetricReq(5, 0, 1, 0, 128)));

  d0.shutdown();
  d1.shutdown();

  EXPECT_FALSE(devD0.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000)).empty());
  // D1 was never contacted -> layer 1 not migrated.
  EXPECT_TRUE(devD1.get(encodeDevice({3, 0}), makeNocAddr(1, 0x9000)).empty());
}

// One decode host's drain fails -> the multiHost reports overall failure even
// though the other host succeeded (partial cluster migration is not success).
TEST(KvMigrationMultiHost, ReportsFailureWhenAHostFails) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0;          // D0 drains fine
  FailingWriteDeviceIo devD1;  // D1 drain always fails
  DecodeNode d0(reg, decode, "D0", "segD0", devD0);
  DecodeNode d1(reg, decode, "D1", "segD1", devD1);

  KvMigrationMultiHostSender multiHost(
      senderEngine, prefillDev, prefill, decode, "P",
      {{"D0", d0.senderCh}, {"D1", d1.senderCh}});
  EXPECT_FALSE(multiHost.migrate(0x9, wholeSlot5()));

  d0.shutdown();
  d1.shutdown();

  // D0 still completed (failure on D1 doesn't roll D0 back).
  EXPECT_EQ(devD0.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000)),
            makeContent(0, 0));
}

// An involved host with no resolved channel makes the multiHost fail (the other
// host is still driven, so the report is comprehensive).
TEST(KvMigrationMultiHost, FailsWhenAnInvolvedHostHasNoChannel) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0;
  DecodeNode d0(reg, decode, "D0", "segD0", devD0);  // only D0 has a channel

  KvMigrationMultiHostSender multiHost(senderEngine, prefillDev, prefill,
                                       decode, "P", {{"D0", d0.senderCh}});
  EXPECT_EQ(multiHost.hostCount(), 1u);
  // Whole slot touches D0 and D1, but D1 is unresolved -> overall failure.
  EXPECT_FALSE(multiHost.migrate(0x2, wholeSlot5()));

  d0.shutdown();
  // D0's layer still landed.
  EXPECT_EQ(devD0.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000)),
            makeContent(0, 0));
}

// End-to-end through the real executor: a Kafka-shaped tt::services::
// MigrationRequest, handed to the real Mooncake-backed IMigrationExecutor,
// drives the whole multi-host data plane and byte-lands on both decode hosts —
// proving the executor is no longer a stub. Same fan-out as
// FansOutToTwoDecodeHosts, but
// triggered through MooncakeMigrationExecutor::execute() (async) instead of a
// direct migrate() call.
TEST(KvMigrationMultiHost, DrivenThroughMooncakeExecutor) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0, devD1;
  DecodeNode d0(reg, decode, "D0", "segD0", devD0);
  DecodeNode d1(reg, decode, "D1", "segD1", devD1);

  KvMigrationMultiHostSender multiHost(
      senderEngine, prefillDev, prefill, decode, "P",
      {{"D0", d0.senderCh}, {"D1", d1.senderCh}});

  MooncakeMigrationExecutor exec(multiHost);

  // The whole-slot request expressed in the scheduler-facing (Kafka) shape;
  // identical fields to wholeSlot5() == symmetricReq(5, 0, 2, 0, 128).
  const tt::services::MigrationRequest api{
      .src_slot = 5,
      .dst_slot = 5,
      .layer_begin = 0,
      .layer_end = 2,
      .src_position_begin = 0,
      .src_position_end = 128,
      .dst_position_begin = 0,
      .dst_position_end = 128,
  };

  std::promise<tt::services::MigrationStatus> done;
  auto fut = done.get_future();
  exec.execute(0x5151, api,
               [&](tt::services::MigrationStatus s) { done.set_value(s); });

  ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_EQ(fut.get(), tt::services::MigrationStatus::SUCCESSFUL);

  d0.shutdown();
  d1.shutdown();

  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    EXPECT_EQ(
        devD0.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
        makeContent(0, p));
    EXPECT_EQ(
        devD1.get(encodeDevice({3, 0}), makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
        makeContent(1, p));
  }
}

}  // namespace
}  // namespace tt::transport
