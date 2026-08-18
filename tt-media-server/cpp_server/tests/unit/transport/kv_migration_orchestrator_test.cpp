// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_orchestrator.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <thread>

#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_bounce_buffer.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_migration_multi_host_sender.hpp"
#include "transport/kv_table_provisioning.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"
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
using test::SpanDeviceIo;
using test::symmetricReq;
using test::wholeSlot5;

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

void expectWholeSlotLanded(const SpanDeviceIo& decodeDev) {
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    const auto l0 = makeContent(0, p);
    const auto l1 = makeContent(1, p);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                            makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK),
              l0);
    EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                            makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK),
              l1);
  }
}

// Build a connected sender/receiver channel pair over crossed pipes.
struct ChannelPair {
  std::shared_ptr<Pipe> ab{std::make_shared<Pipe>()};  // sender -> receiver
  std::shared_ptr<Pipe> ba{std::make_shared<Pipe>()};  // receiver -> sender
  std::shared_ptr<BlockingFakeTransport> senderTp{
      std::make_shared<BlockingFakeTransport>(ba, ab)};
  std::shared_ptr<BlockingFakeTransport> receiverTp{
      std::make_shared<BlockingFakeTransport>(ab, ba)};
  KvControlChannel senderCh{senderTp};
  KvControlChannel receiverCh{receiverTp};
};

// The bounce orchestrators drive a complete migration over the channel: the
// receiver runs on its own thread reacting to
// BounceReady/WindowReady/DoneMarker, the sender drives windows + credits, and
// the bytes land on the decode device. A single-slot bounce buffer forces the
// multi-window credit handshake over the wire.
TEST(KvMigrationOrchestrator, DrivesWindowedMigrationOverChannel) {
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

  const BounceGeometry geo{/*section_count=*/1,
                           /*section_size=*/K_CHUNK};  // many windows
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  ChannelPair cp;
  KvMigrationSender senderOrch(cp.senderCh, sender);
  KvMigrationReceiver receiverOrch(cp.receiverCh, receiver);

  std::thread receiverThread([&] { receiverOrch.run(); });
  EXPECT_TRUE(senderOrch.migrate(0x5151, wholeSlot5()));
  closePipe(cp.ab);
  receiverThread.join();

  expectWholeSlotLanded(decodeDev);
}

// A position shift driven over the channel: dst coordinates travel in
// BeginMigration, the sender's descriptors carry the dst device addresses, and
// src [0,64) lands at dst [64,128).
TEST(KvMigrationOrchestrator, DrivesPositionShiftOverChannel) {
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

  ChannelPair cp;
  KvMigrationSender senderOrch(cp.senderCh, sender);
  KvMigrationReceiver receiverOrch(cp.receiverCh, receiver);

  std::thread receiverThread([&] { receiverOrch.run(); });
  EXPECT_TRUE(
      senderOrch.migrate(0x5417, asymmetricReq(5, 5, 0, 2, 0, 64, 64, 128)));
  closePipe(cp.ab);
  receiverThread.join();

  EXPECT_EQ(decodeDev.get(encodeDevice({2, 0}),
                          makeNocAddr(0, 0x8000 + 2 * K_CHUNK), K_CHUNK),
            makeContent(0, 0));
  EXPECT_EQ(decodeDev.get(encodeDevice({2, 2}),
                          makeNocAddr(1, 0x9000 + 3 * K_CHUNK), K_CHUNK),
            makeContent(1, 32));
}

// A failed device drain is reported via WindowAck.ok = false, so migrate()
// fails rather than treating a corrupt cache as success.
TEST(KvMigrationOrchestrator, ReportsDrainFailure) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto receiverEngine = std::make_shared<FakeTransferEngine>(reg, "D");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  class FailWriteDeviceIo : public FakeDeviceIo {
   public:
    bool write(LocalDeviceId, NocAddr, const void*, std::size_t) override {
      return false;
    }
  } decodeDev;

  const BounceGeometry geo{4, 256};
  MooncakeKvReceiver receiver(receiverEngine, decodeDev, "D", geo);
  MooncakeKvSender sender(senderEngine, prefillDev, prefill, decode, "P", "D");

  ChannelPair cp;
  KvMigrationSender senderOrch(cp.senderCh, sender);
  KvMigrationReceiver receiverOrch(cp.receiverCh, receiver);

  std::thread receiverThread([&] { receiverOrch.run(); });
  EXPECT_FALSE(senderOrch.migrate(0x7373, wholeSlot5()));
  closePipe(cp.ab);
  receiverThread.join();
}

// Dry-run mode (null receiver): the orchestrator still serves TABLE_EXCHANGE so
// prefill can finish init, but a migration request is rejected — BeginMigration
// yields BounceReady{ok=false} with no segment, and no device or bounce buffer
// is touched. Covers the control-only path the decode worker uses without
// hardware (KV_MIGRATION_MODE=dry-run).
TEST(KvMigrationOrchestrator, DryRunServesTableExchangeAndRejectsMigration) {
  ChannelPair cp;

  const std::vector<uint8_t> localTable{4, 5, 6};
  KvMigrationReceiver receiverOrch{
      cp.receiverCh, static_cast<MooncakeKvReceiver*>(nullptr),
      std::make_shared<const std::vector<uint8_t>>(localTable)};

  // TABLE_EXCHANGE is served even without a receiver.
  KvControlMessage exchange;
  exchange.type = KvControlType::TABLE_EXCHANGE;
  exchange.role = static_cast<uint8_t>(TableExchangeRole::Sender);
  exchange.table_blob = {1, 2, 3};
  ASSERT_TRUE(cp.senderCh.send(exchange));
  ASSERT_TRUE(receiverOrch.serveOne());

  const auto exchangeReply = cp.senderCh.receive();
  ASSERT_TRUE(exchangeReply.has_value());
  EXPECT_EQ(exchangeReply->type, KvControlType::TABLE_EXCHANGE);
  EXPECT_EQ(exchangeReply->table_blob, localTable);

  // A migration request is rejected: BounceReady{ok=false}, empty segment.
  KvControlMessage begin;
  begin.type = KvControlType::BEGIN_MIGRATION;
  begin.uuid = 42;
  ASSERT_TRUE(cp.senderCh.send(begin));
  ASSERT_TRUE(receiverOrch.serveOne());

  const auto ready = cp.senderCh.receive();
  ASSERT_TRUE(ready.has_value());
  EXPECT_EQ(ready->type, KvControlType::BOUNCE_READY);
  EXPECT_EQ(ready->uuid, 42u);
  EXPECT_FALSE(ready->ok);
  EXPECT_TRUE(ready->segment_name.empty());
}

// --- Bounce n->m multi-host fan-out (C2) -------------------------------------

// One decode host for the multi-host bounce buffer tests: its own
// engine/segment, a control-channel pair, a bounce receiver (holds NO table —
// drains from window descriptors), and a thread running the bounce receiver
// orchestrator.
struct BounceDecodeNode {
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

  BounceDecodeNode(const std::shared_ptr<FakeRegistry>& reg,
                   const std::string& seg, SpanDeviceIo& dev,
                   BounceGeometry geo) {
    engine = std::make_shared<FakeTransferEngine>(reg, seg);
    senderTp = std::make_shared<BlockingFakeTransport>(/*in=*/ba, /*out=*/ab);
    receiverTp = std::make_shared<BlockingFakeTransport>(/*in=*/ab, /*out=*/ba);
    senderCh = std::make_shared<KvControlChannel>(senderTp);
    receiverCh = std::make_unique<KvControlChannel>(receiverTp);
    receiver = std::make_unique<MooncakeKvReceiver>(engine, dev, seg, geo);
    orch = std::make_unique<KvMigrationReceiver>(*receiverCh, *receiver);
    thread = std::thread([this] { orch->run(); });
  }
  void shutdown() {
    closePipe(ab);
    if (thread.joinable()) thread.join();
  }
  ~BounceDecodeNode() { shutdown(); }
};

// A whole-slot bounce migration fans out to TWO decode hosts: layer 0 lands on
// D0, layer 1 on D1, each over its own bounce-buffer segment + control channel,
// byte-verified.
TEST(KvMigrationMultiHost, FansOutToTwoDecodeHosts) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");

  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  // layer 0 -> host "D0" (mesh 2), layer 1 -> host "D1" (mesh 3).
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  SpanDeviceIo devD0, devD1;
  const BounceGeometry geo{/*section_count=*/1,
                           /*section_size=*/K_CHUNK};  // multi-window
  BounceDecodeNode d0(reg, "segD0", devD0, geo);
  BounceDecodeNode d1(reg, "segD1", devD1, geo);

  KvMigrationMultiHostSender multi(senderEngine, prefillDev, prefill, decode,
                                   "P",
                                   {{"D0", d0.senderCh}, {"D1", d1.senderCh}});
  EXPECT_EQ(multi.hostCount(), 2u);
  EXPECT_TRUE(multi.migrate(0x5151, wholeSlot5()));

  d0.shutdown();
  d1.shutdown();

  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    // D0 holds layer 0 on both its replicas; D1 holds layer 1.
    EXPECT_EQ(devD0.get(encodeDevice({2, 0}),
                        makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK),
              makeContent(0, p));
    EXPECT_EQ(devD0.get(encodeDevice({2, 1}),
                        makeNocAddr(0, 0x8000 + idx * K_CHUNK), K_CHUNK),
              makeContent(0, p));
    EXPECT_EQ(devD1.get(encodeDevice({3, 0}),
                        makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK),
              makeContent(1, p));
    EXPECT_EQ(devD1.get(encodeDevice({3, 1}),
                        makeNocAddr(1, 0x9000 + idx * K_CHUNK), K_CHUNK),
              makeContent(1, p));
  }
}

// The multi-host bounce sender drives ONLY the hosts the request touches: a
// layer-0-only request lands on D0 and never contacts D1.
TEST(KvMigrationMultiHost, DrivesOnlyHostsInRequest) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedWholeSlot(prefillDev);
  SpanDeviceIo devD0, devD1;
  const BounceGeometry geo{4, 256};
  BounceDecodeNode d0(reg, "segD0", devD0, geo);
  BounceDecodeNode d1(reg, "segD1", devD1, geo);

  KvMigrationMultiHostSender multi(senderEngine, prefillDev, prefill, decode,
                                   "P",
                                   {{"D0", d0.senderCh}, {"D1", d1.senderCh}});

  // Layer [0,1) lives only on D0.
  EXPECT_TRUE(multi.migrate(0x1, symmetricReq(5, 0, 1, 0, 128)));

  d0.shutdown();
  d1.shutdown();

  EXPECT_FALSE(
      devD0.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000), K_CHUNK).empty());
  // D1 was never contacted -> layer 1 not migrated.
  EXPECT_TRUE(
      devD1.get(encodeDevice({3, 0}), makeNocAddr(1, 0x9000), K_CHUNK).empty());
}

}  // namespace
}  // namespace tt::transport
