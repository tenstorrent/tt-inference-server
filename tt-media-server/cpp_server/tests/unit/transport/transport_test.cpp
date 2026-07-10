// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "transport/device_dram_storage_backend.hpp"
#include "transport/host_dram_storage_backend.hpp"
#include "transport/i_storage_backend.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/mooncake_migration_worker.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/peer_discovery_service.hpp"
#include "transport/peer_table_exchange.hpp"
#include "transport/transfer_types.hpp"
#include "transport/umd_device_access.hpp"

namespace tt::transport {
namespace {

// NocAddr packs channel and channel-local offset; the helpers must round-trip.
TEST(TransferTypes, NocAddrRoundTrips) {
  const uint32_t channel = 3;
  const uint32_t local = 0x1000;
  const NocAddr addr = makeNocAddr(channel, local);
  EXPECT_EQ(nocChannel(addr), channel);
  EXPECT_EQ(nocLocalAddr(addr), local);
}

// Storage mechanism: backends report their medium through the interface.
TEST(StorageBackend, ReportMediumThroughInterface) {
  std::unique_ptr<IStorageBackend> host =
      std::make_unique<HostDramStorageBackend>();
  EXPECT_EQ(host->medium(), StorageMedium::HOST_DRAM);

  auto device = std::make_unique<DeviceDramStorageBackend>(
      std::make_shared<UmdDeviceAccess>(/*device_id=*/0));
  EXPECT_EQ(device->medium(), StorageMedium::DEVICE_DRAM);
}

// The host-DRAM backend stages bytes via memcpy: writeFrom then readInto a
// separate host region must round-trip the payload.
TEST(HostDramStorageBackend, ReadWriteRoundTrip) {
  HostDramStorageBackend backend;

  std::vector<uint8_t> store(64, 0);
  std::vector<uint8_t> src(store.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i);
  }

  // `addr` is a host virtual address for this backend.
  const auto storeAddr = reinterpret_cast<uint64_t>(store.data());
  EXPECT_TRUE(backend.writeFrom(storeAddr, src.data(), src.size()));
  EXPECT_EQ(store, src);

  std::vector<uint8_t> dst(store.size(), 0);
  EXPECT_TRUE(backend.readInto(storeAddr, dst.size(), dst.data()));
  EXPECT_EQ(dst, src);
}

// Zero-length transfers are a no-op success; null addr/buffer is rejected.
TEST(HostDramStorageBackend, ZeroLengthSucceedsNullFails) {
  HostDramStorageBackend backend;
  std::vector<uint8_t> buffer(8, 0);
  const auto addr = reinterpret_cast<uint64_t>(buffer.data());

  EXPECT_TRUE(backend.readInto(addr, 0, buffer.data()));
  EXPECT_TRUE(backend.writeFrom(addr, buffer.data(), 0));

  EXPECT_FALSE(backend.readInto(/*addr=*/0, buffer.size(), buffer.data()));
  EXPECT_FALSE(backend.writeFrom(addr, /*hostBuffer=*/nullptr, buffer.size()));
}

// The device-DRAM custom backend delegates to UMD; placeholder reports failure.
#ifndef USE_METAL_CPP_LIB
// Without the real UMD backend in the build, the device-DRAM storage methods
// report failure (no device) without crashing. With USE_METAL_CPP_LIB compiled
// in, readInto/writeFrom touch real device DRAM, so that path is exercised by
// the integration tests instead.
TEST(DeviceDramStorageBackend, MethodsReportNotImplemented) {
  DeviceDramStorageBackend backend(
      std::make_shared<UmdDeviceAccess>(/*device_id=*/0));
  std::vector<uint8_t> buffer(64, 0);
  const auto addr = makeNocAddr(/*channel=*/0, /*local_addr=*/0);
  EXPECT_FALSE(backend.readInto(addr, buffer.size(), buffer.data()));
  EXPECT_FALSE(backend.writeFrom(addr, buffer.data(), buffer.size()));
}
#endif  // USE_METAL_CPP_LIB

// The Transfer Engine composes a storage backend and reports its medium, and
// exposes that same backend through storage() for the bounce-buffer flow.
TEST(MooncakeTransferEngine, ComposesStorageBackend) {
  auto storage = std::make_shared<DeviceDramStorageBackend>(
      std::make_shared<UmdDeviceAccess>(/*device_id=*/0));
  std::unique_ptr<ITransferEngine> engine =
      std::make_unique<MooncakeTransferEngine>(storage);
  ASSERT_NE(engine, nullptr);
  EXPECT_EQ(engine->storageMedium(), StorageMedium::DEVICE_DRAM);
  EXPECT_EQ(engine->storage(), storage);
}

#ifndef TT_TRANSPORT_WITH_MOONCAKE
// Without the Mooncake backend in the build, the transport methods report
// failure (not a live engine) without crashing. With Mooncake compiled in,
// init() touches the network/metadata service, so that path is exercised by
// the integration tests instead.
TEST(MooncakeTransferEngine, MethodsReportFailureWithoutMooncake) {
  MooncakeTransferEngine engine{std::make_shared<HostDramStorageBackend>()};
  EXPECT_FALSE(engine.init(EngineConfig{}));

  std::vector<uint8_t> buffer(64, 0);
  EXPECT_FALSE(engine.registerLocalMemory(buffer.data(), buffer.size()));
  EXPECT_EQ(engine.openSegment("peer"), K_INVALID_SEGMENT);

  TransferRequest request{TransferOp::WRITE, buffer.data(), K_INVALID_SEGMENT,
                          0, buffer.size()};
  EXPECT_EQ(engine.submitAndWait(request).state, TransferState::FAILED);
}
#endif  // TT_TRANSPORT_WITH_MOONCAKE

#ifndef USE_METAL_CPP_LIB
// The UMD access wrapper constructs and its placeholder I/O reports failure.
// With USE_METAL_CPP_LIB compiled in, read/write touch real device DRAM, so
// that path is exercised by the integration tests instead.
TEST(UmdDeviceAccess, MethodsReportNotImplemented) {
  UmdDeviceAccess device(/*device_id=*/0);
  std::vector<uint8_t> buffer(64, 0);
  const NocAddr addr = makeNocAddr(/*channel=*/0, /*local_addr=*/0);
  EXPECT_FALSE(device.read(addr, buffer.size(), buffer.data()));
  EXPECT_FALSE(device.write(addr, buffer.data(), buffer.size()));
}
#endif  // USE_METAL_CPP_LIB

// The migration worker's storage-staging steps (write on sender, verify on
// receiver) work end-to-end against a host-DRAM backend standing in for device
// DRAM — no live transport needed.
TEST(MooncakeMigrationWorker, StorageStagingRoundTrips) {
  auto storage = std::make_shared<HostDramStorageBackend>();
  auto engine = std::make_shared<MooncakeTransferEngine>(storage);

  // A host region acts as the "device DRAM" the worker stages to/from.
  std::vector<uint8_t> deviceRegion(64, 0);
  const auto addr = reinterpret_cast<uint64_t>(deviceRegion.data());

  MigrationWorkerConfig senderCfg;
  senderCfg.role = MigrationRole::SENDER;
  senderCfg.peer_segment_name = "receiver";
  senderCfg.device_addr = addr;
  senderCfg.tensor_bytes = deviceRegion.size();
  MooncakeMigrationWorker sender(senderCfg, engine, nullptr);

  const std::vector<uint8_t> tensor(deviceRegion.size(), 0xAB);
  EXPECT_TRUE(sender.writeTensorOnSender(tensor));
  EXPECT_EQ(deviceRegion, tensor);  // tensor landed in "device DRAM"

  MigrationWorkerConfig receiverCfg = senderCfg;
  receiverCfg.role = MigrationRole::RECEIVER;
  MooncakeMigrationWorker receiver(receiverCfg, engine, nullptr);
  EXPECT_TRUE(receiver.verifyTensorOnReceiver(tensor));

  const std::vector<uint8_t> wrong(deviceRegion.size(), 0x00);
  EXPECT_FALSE(receiver.verifyTensorOnReceiver(wrong));
}

// Role guards and the transport hop: a sender cannot verify, a receiver cannot
// write/transfer, and transferToReceiver needs a live (init'd) engine — without
// one it reports failure rather than crashing.
TEST(MooncakeMigrationWorker, RoleGuardsAndTransportNeedsLiveEngine) {
  auto storage = std::make_shared<HostDramStorageBackend>();
  auto engine = std::make_shared<MooncakeTransferEngine>(storage);

  std::vector<uint8_t> deviceRegion(64, 0);
  MigrationWorkerConfig config;
  config.role = MigrationRole::SENDER;
  config.peer_segment_name = "receiver";
  config.device_addr = reinterpret_cast<uint64_t>(deviceRegion.data());
  config.tensor_bytes = deviceRegion.size();

  MooncakeMigrationWorker sender(config, engine, nullptr);
  const std::vector<uint8_t> tensor(config.tensor_bytes, 0xAB);
  // Sender cannot run the receiver step.
  EXPECT_FALSE(sender.verifyTensorOnReceiver(tensor));
  // The transport hop has no init'd engine / peer segment here.
  EXPECT_FALSE(sender.transferToReceiver());

  config.role = MigrationRole::RECEIVER;
  MooncakeMigrationWorker receiver(config, engine, nullptr);
  // Receiver cannot run the sender steps.
  EXPECT_FALSE(receiver.writeTensorOnSender(tensor));
  EXPECT_FALSE(receiver.transferToReceiver());
}

TEST(MooncakeMigrationWorker, OwnsLayerHonorsConfiguredSpan) {
  MigrationWorkerConfig unset;
  MooncakeMigrationWorker ownsAll(unset, nullptr, nullptr);
  EXPECT_TRUE(ownsAll.ownsLayer(0));
  EXPECT_TRUE(ownsAll.ownsLayer(999));

  MigrationWorkerConfig sharded;
  sharded.layer_start = 4;
  sharded.layer_end = 8;
  MooncakeMigrationWorker shard(sharded, nullptr, nullptr);
  EXPECT_FALSE(shard.ownsLayer(3));
  EXPECT_TRUE(shard.ownsLayer(4));
  EXPECT_TRUE(shard.ownsLayer(7));
  EXPECT_FALSE(shard.ownsLayer(8));
}

// ---------------------------------------------------------------------------
// Discovery + bring-up unit tests (#4294)
//
// These exercise PeerDiscoveryService and MooncakeMigrationWorker::bringUp
// against a programmable fake engine — no Mooncake, no metadata service, no
// network — so
// they run in any build/CI. The fake records call order and lets each test
// dictate exactly when (or whether) a peer resolves.
// ---------------------------------------------------------------------------

// A scriptable ITransferEngine: init/register results are configurable, and
// each peer resolves only once openSegment has been called more than
// `resolveAfterMisses[name]` times (absent => never resolves). Every call is
// appended to callLog so tests can assert ordering (e.g. register before open).
class FakeTransferEngine : public ITransferEngine {
 public:
  bool initResult = true;
  bool registerResult = true;
  std::map<std::string, int> resolveAfterMisses;  ///< name -> misses before hit

  std::vector<std::string> callLog;
  int registerCount = 0;
  int unregisterCount = 0;
  std::map<std::string, int> openAttempts;

  StorageMedium storageMedium() const override {
    return StorageMedium::HOST_DRAM;
  }
  std::shared_ptr<IStorageBackend> storage() const override { return nullptr; }

  bool init(const EngineConfig&) override {
    callLog.emplace_back("init");
    return initResult;
  }
  bool registerLocalMemory(void*, std::size_t) override {
    callLog.emplace_back("register");
    ++registerCount;
    return registerResult;
  }
  bool unregisterLocalMemory(void*) override {
    callLog.emplace_back("unregister");
    ++unregisterCount;
    return true;
  }
  SegmentHandle openSegment(const std::string& name) override {
    callLog.emplace_back("open:" + name);
    const int attempt = ++openAttempts[name];
    const auto it = resolveAfterMisses.find(name);
    if (it == resolveAfterMisses.end()) return K_INVALID_SEGMENT;
    return attempt > it->second ? static_cast<SegmentHandle>(attempt)
                                : K_INVALID_SEGMENT;
  }
  SegmentHandle refreshSegment(const std::string& name) override {
    callLog.emplace_back("refresh:" + name);
    return static_cast<SegmentHandle>(1);
  }
  TransferStatus submitAndWait(const TransferRequest&) override {
    return {TransferState::COMPLETED, 0};
  }
};

// Fake that models TE table exchange: WRITEs copy into a peer's registered
// recv base (keyed by SegmentHandle). Enough to unit-test PeerTableExchange
// without Mooncake.
class TableExchangeFakeEngine : public ITransferEngine {
 public:
  std::map<SegmentHandle, std::uint8_t*> peerRecvBase;
  std::vector<void*> registered;

  StorageMedium storageMedium() const override {
    return StorageMedium::HOST_DRAM;
  }
  std::shared_ptr<IStorageBackend> storage() const override { return nullptr; }
  bool init(const EngineConfig&) override { return true; }
  bool registerLocalMemory(void* addr, std::size_t) override {
    registered.push_back(addr);
    return true;
  }
  bool unregisterLocalMemory(void* addr) override {
    const auto it = std::find(registered.begin(), registered.end(), addr);
    if (it != registered.end()) registered.erase(it);
    return true;
  }
  void* firstRegisteredLocalBuffer() const override {
    return registered.empty() ? nullptr : registered.front();
  }
  std::size_t registeredLocalBufferCount() const override {
    return registered.size();
  }
  SegmentHandle openSegment(const std::string&) override {
    return K_INVALID_SEGMENT;
  }
  SegmentHandle refreshSegment(const std::string&) override {
    return K_INVALID_SEGMENT;
  }
  TransferStatus submitAndWait(const TransferRequest& req) override {
    if (req.op != TransferOp::WRITE) {
      return {TransferState::FAILED, 0};
    }
    const auto it = peerRecvBase.find(req.target);
    if (it == peerRecvBase.end() || it->second == nullptr) {
      return {TransferState::FAILED, 0};
    }
    std::memcpy(it->second + req.target_offset, req.local_addr, req.length);
    // Release so waitForFlag's acquire load observes the flag (and prior body).
    std::atomic_thread_fence(std::memory_order_release);
    return {TransferState::COMPLETED, 0};
  }
};

TEST(PeerTableExchange, Fnv1aIsStable) {
  const std::uint8_t data[] = {1, 2, 3, 4};
  EXPECT_EQ(PeerTableExchange::fnv1a(data, 4),
            PeerTableExchange::fnv1a(data, 4));
  EXPECT_NE(PeerTableExchange::fnv1a(data, 4),
            PeerTableExchange::fnv1a(data, 3));
}

// 1:1 exchange into each other's single slot.
TEST(PeerTableExchange, RoundTripTwoPeers) {
  PeerTableExchangeConfig cfg;
  cfg.timeoutSec = 2;
  cfg.pollIntervalMs = 1;
  cfg.maxTableBytes = 64;
  PeerTableExchange xchg(cfg);

  const std::vector<std::uint8_t> tableA = {10, 20, 30, 40};
  const std::vector<std::uint8_t> tableB = {50, 60, 70, 80, 90};

  std::vector<std::uint8_t> recvA(xchg.requiredRecvBytes(1), 0);
  std::vector<std::uint8_t> recvB(xchg.requiredRecvBytes(1), 0);

  constexpr SegmentHandle kHandleA = 1;
  constexpr SegmentHandle kHandleB = 2;

  TableExchangeFakeEngine engineA;
  engineA.peerRecvBase[kHandleB] = recvB.data();
  TableExchangeFakeEngine engineB;
  engineB.peerRecvBase[kHandleA] = recvA.data();

  using Slot = PeerTableExchange::PeerSlot;
  std::optional<std::map<std::string, std::vector<std::uint8_t>>> gotA;
  std::optional<std::map<std::string, std::vector<std::uint8_t>>> gotB;
  std::thread tB([&] {
    gotB = xchg.exchange(
        engineB, {{"peer-a", Slot{kHandleA, /*local=*/0, /*remote=*/0}}},
        "peer-b", tableB, recvB.data());
  });
  gotA = xchg.exchange(engineA,
                       {{"peer-b", Slot{kHandleB, /*local=*/0, /*remote=*/0}}},
                       "peer-a", tableA, recvA.data());
  tB.join();

  ASSERT_TRUE(gotA.has_value());
  ASSERT_TRUE(gotB.has_value());
  EXPECT_EQ(gotA->at("peer-b"), tableB);
  EXPECT_EQ(gotB->at("peer-a"), tableA);
}

// Prefill with two decode peers: each decode WRITEs into a distinct slot —
// concurrent fan-in must not corrupt either blob.
TEST(PeerTableExchange, MultiPeerFanInIsolatedSlots) {
  PeerTableExchangeConfig cfg;
  cfg.timeoutSec = 2;
  cfg.pollIntervalMs = 1;
  cfg.maxTableBytes = 64;
  PeerTableExchange xchg(cfg);

  const std::vector<std::uint8_t> tableP = {1, 1, 1};
  const std::vector<std::uint8_t> tableD0 = {2, 2, 2, 2};
  const std::vector<std::uint8_t> tableD1 = {3, 3, 3, 3, 3};

  // Prefill peers = [decode-0, decode-1] (sorted) → 2 slots.
  // Each decode peers only prefill → 1 slot; prefill is index 0 in that list.
  std::vector<std::uint8_t> recvP(xchg.requiredRecvBytes(2), 0);
  std::vector<std::uint8_t> recvD0(xchg.requiredRecvBytes(1), 0);
  std::vector<std::uint8_t> recvD1(xchg.requiredRecvBytes(1), 0);

  constexpr SegmentHandle kPrefill = 10;
  constexpr SegmentHandle kD0 = 20;
  constexpr SegmentHandle kD1 = 21;

  TableExchangeFakeEngine engP;
  engP.peerRecvBase[kD0] = recvD0.data();
  engP.peerRecvBase[kD1] = recvD1.data();
  TableExchangeFakeEngine engD0;
  engD0.peerRecvBase[kPrefill] = recvP.data();
  TableExchangeFakeEngine engD1;
  engD1.peerRecvBase[kPrefill] = recvP.data();

  using Slot = PeerTableExchange::PeerSlot;
  std::optional<std::map<std::string, std::vector<std::uint8_t>>> gotP, gotD0,
      gotD1;

  std::thread t0([&] {
    gotD0 = xchg.exchange(
        engD0, {{"prefill", Slot{kPrefill, /*local=*/0, /*remote=*/0}}},
        "decode-0", tableD0, recvD0.data());
  });
  std::thread t1([&] {
    gotD1 = xchg.exchange(
        engD1, {{"prefill", Slot{kPrefill, /*local=*/0, /*remote=*/1}}},
        "decode-1", tableD1, recvD1.data());
  });
  gotP = xchg.exchange(engP,
                       {{"decode-0", Slot{kD0, /*local=*/0, /*remote=*/0}},
                        {"decode-1", Slot{kD1, /*local=*/1, /*remote=*/0}}},
                       "prefill", tableP, recvP.data());
  t0.join();
  t1.join();

  ASSERT_TRUE(gotP.has_value());
  ASSERT_TRUE(gotD0.has_value());
  ASSERT_TRUE(gotD1.has_value());
  EXPECT_EQ(gotP->at("decode-0"), tableD0);
  EXPECT_EQ(gotP->at("decode-1"), tableD1);
  EXPECT_EQ(gotD0->at("prefill"), tableP);
  EXPECT_EQ(gotD1->at("prefill"), tableP);
}

// register → unregister → register must make the new buffer buffers[0].
TEST(PeerTableExchange, RegisterUnregisterRestoresBuffers0) {
  TableExchangeFakeEngine engine;
  std::uint8_t recvSlot[32]{};
  std::uint8_t mirror[64]{};
  ASSERT_TRUE(engine.registerLocalMemory(recvSlot, sizeof(recvSlot)));
  EXPECT_EQ(engine.firstRegisteredLocalBuffer(), recvSlot);
  ASSERT_TRUE(engine.registerLocalMemory(mirror, sizeof(mirror)));
  EXPECT_EQ(engine.firstRegisteredLocalBuffer(), recvSlot);
  EXPECT_EQ(engine.registeredLocalBufferCount(), 2u);
  ASSERT_TRUE(engine.unregisterLocalMemory(recvSlot));
  EXPECT_EQ(engine.firstRegisteredLocalBuffer(), mirror);
  ASSERT_TRUE(engine.unregisterLocalMemory(mirror));
  EXPECT_EQ(engine.registeredLocalBufferCount(), 0u);
  ASSERT_TRUE(engine.registerLocalMemory(mirror, sizeof(mirror)));
  EXPECT_EQ(engine.firstRegisteredLocalBuffer(), mirror);
  EXPECT_EQ(engine.registeredLocalBufferCount(), 1u);
}

TEST(PeerTableExchange, RejectsOversizedLocalBlob) {
  PeerTableExchangeConfig cfg;
  cfg.maxTableBytes = 4;
  PeerTableExchange xchg(cfg);
  TableExchangeFakeEngine engine;
  std::vector<std::uint8_t> recv(xchg.requiredRecvBytes(1), 0);
  std::vector<std::uint8_t> big(8, 1);
  using Slot = PeerTableExchange::PeerSlot;
  EXPECT_FALSE(
      xchg.exchange(engine, {{"p", Slot{1, 0, 0}}}, "self", big, recv.data())
          .has_value());
}

TEST(PeerTableExchange, EmptyPeersIsSuccess) {
  PeerTableExchange xchg;
  TableExchangeFakeEngine engine;
  std::vector<std::uint8_t> blob = {1};
  const auto got = xchg.exchange(engine, {}, "self", blob, nullptr);
  ASSERT_TRUE(got.has_value());
  EXPECT_TRUE(got->empty());
}

// Fast tunables so the polling path runs in milliseconds, not seconds.
PeerDiscoveryConfig fastDiscovery(int timeoutSec) {
  return PeerDiscoveryConfig{/*poll_interval_ms=*/1,
                             /*timeout_sec=*/timeoutSec};
}

std::shared_ptr<PeerDiscoveryService> fastDiscoveryService(int timeoutSec) {
  return std::make_shared<PeerDiscoveryService>(fastDiscovery(timeoutSec));
}

TEST(PeerDiscoveryService, ResolvesAllPeersInOneSweep) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"a", 0}, {"b", 0}, {"c", 0}};
  PeerDiscoveryService discovery(fastDiscovery(5));

  const auto resolved = discovery.discover(engine, {"a", "b", "c"});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->size(), 3u);
  for (const auto& name : {"a", "b", "c"}) {
    EXPECT_EQ(engine.openAttempts[name], 1);
    EXPECT_NE(resolved->at(name), K_INVALID_SEGMENT);
  }
}

TEST(PeerDiscoveryService, RetriesOnlyUnresolvedUntilTheyAppear) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"ready", 0}, {"late", 2}};
  PeerDiscoveryService discovery(fastDiscovery(5));

  const auto resolved = discovery.discover(engine, {"ready", "late"});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->size(), 2u);
  EXPECT_EQ(engine.openAttempts["ready"], 1);
  EXPECT_GE(engine.openAttempts["late"], 3);
}

TEST(PeerDiscoveryService, TimesOutWhenAPeerNeverResolves) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"present", 0}};
  PeerDiscoveryService discovery(fastDiscovery(1));

  const auto resolved = discovery.discover(engine, {"present", "ghost"});
  EXPECT_FALSE(resolved.has_value());
  EXPECT_GE(engine.openAttempts["ghost"], 1);
}

TEST(PeerDiscoveryService, NoPeersResolvesToEmptyMap) {
  FakeTransferEngine engine;
  PeerDiscoveryService discovery(fastDiscovery(5));

  const auto resolved = discovery.discover(engine, {});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_TRUE(resolved->empty());
  EXPECT_TRUE(engine.callLog.empty());
}

TEST(PeerDiscoveryService, DeduplicatesRepeatedPeerNames) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"a", 0}, {"b", 0}};
  PeerDiscoveryService discovery(fastDiscovery(1));

  const auto resolved = discovery.discover(engine, {"a", "a", "b", "a"});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->size(), 2u);
  EXPECT_EQ(engine.openAttempts["a"], 1);
}

TEST(PeerDiscoveryService, IgnoresEmptyPeerNames) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"real", 0}};
  PeerDiscoveryService discovery(fastDiscovery(1));

  const auto resolved = discovery.discover(engine, {"", "real", ""});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->size(), 1u);
  EXPECT_EQ(engine.openAttempts.count(""), 0u);
}

TEST(PeerDiscoveryService, CancelTokenAbortsPromptly) {
  FakeTransferEngine engine;
  PeerDiscoveryService discovery(fastDiscovery(600));
  std::atomic<bool> cancel{true};

  const auto resolved = discovery.discover(engine, {"ghost"}, &cancel);
  EXPECT_FALSE(resolved.has_value());
  EXPECT_EQ(engine.openAttempts["ghost"], 1);
}

TEST(PeerDiscoveryService, ZeroTimeoutTriesExactlyOnce) {
  FakeTransferEngine engine;
  PeerDiscoveryService discovery(fastDiscovery(0));

  const auto resolved = discovery.discover(engine, {"ghost"});
  EXPECT_FALSE(resolved.has_value());
  EXPECT_EQ(engine.openAttempts["ghost"], 1);
}

// Bring-up happy path: the worker inits, publishes its pool, then discovers its
// peer — and crucially registers BEFORE opening any peer segment, so peers can
// resolve it in return (the #4294 ordering invariant).
TEST(MooncakeMigrationWorker, BringUpRegistersBeforeConnecting) {
  auto engine = std::make_shared<FakeTransferEngine>();
  engine->resolveAfterMisses = {{"peer-0", 0}};

  MigrationWorkerConfig cfg;
  cfg.metadata_uri = "fake://meta";
  cfg.segment_name = "self";
  cfg.host_dram_bytes = 4096;
  cfg.peer_segment_names = {"peer-0"};

  MooncakeMigrationWorker worker{cfg, engine, fastDiscoveryService(5)};
  ASSERT_TRUE(worker.bringUp());
  EXPECT_EQ(worker.peers().size(), 1u);
  EXPECT_EQ(engine->registerCount, 1);

  // register must appear before the first peer open in the recorded order.
  const auto& log = engine->callLog;
  const auto reg = std::find(log.begin(), log.end(), "register");
  const auto open = std::find(log.begin(), log.end(), "open:peer-0");
  ASSERT_NE(reg, log.end());
  ASSERT_NE(open, log.end());
  EXPECT_LT(reg - log.begin(), open - log.begin());
}

// Guard rails: a null engine and a zero-sized pool both fail before any
// resource is touched.
TEST(MooncakeMigrationWorker, BringUpRejectsBadConfig) {
  MigrationWorkerConfig cfg;
  cfg.host_dram_bytes = 4096;
  cfg.peer_segment_names = {"peer-0"};

  MooncakeMigrationWorker noEngine(cfg, nullptr, fastDiscoveryService(5));
  EXPECT_FALSE(noEngine.bringUp());

  auto engine = std::make_shared<FakeTransferEngine>();
  MigrationWorkerConfig zeroPool = cfg;
  zeroPool.host_dram_bytes = 0;
  MooncakeMigrationWorker worker{zeroPool, engine, fastDiscoveryService(5)};
  EXPECT_FALSE(worker.bringUp());
  EXPECT_EQ(engine->registerCount, 0);  // nothing allocated/published
}

// Fail-fast: if engine init fails we never register; if register fails we never
// try to open a peer segment.
TEST(MooncakeMigrationWorker, BringUpStopsAtFirstFailedPhase) {
  MigrationWorkerConfig cfg;
  cfg.host_dram_bytes = 4096;
  cfg.peer_segment_names = {"peer-0"};

  auto initFails = std::make_shared<FakeTransferEngine>();
  initFails->initResult = false;
  MooncakeMigrationWorker w1(cfg, initFails, fastDiscoveryService(5));
  EXPECT_FALSE(w1.bringUp());
  EXPECT_EQ(initFails->registerCount, 0);

  auto regFails = std::make_shared<FakeTransferEngine>();
  regFails->registerResult = false;
  MooncakeMigrationWorker w2(cfg, regFails, fastDiscoveryService(5));
  EXPECT_FALSE(w2.bringUp());
  EXPECT_EQ(regFails->openAttempts.count("peer-0"), 0u);  // never reached
}

// If discovery times out, bring-up unwinds: the published pool is unregistered
// so we don't leave a half-initialised worker advertised to the cluster.
TEST(MooncakeMigrationWorker, BringUpUnwindsWhenDiscoveryTimesOut) {
  auto engine = std::make_shared<FakeTransferEngine>();
  // "ghost" never resolves; timeout 0 makes resolveAll give up immediately.
  MigrationWorkerConfig cfg;
  cfg.host_dram_bytes = 4096;
  cfg.segment_name = "self";
  cfg.peer_segment_names = {"ghost"};

  MooncakeMigrationWorker worker{cfg, engine, fastDiscoveryService(0)};
  EXPECT_FALSE(worker.bringUp());
  EXPECT_EQ(engine->registerCount, 1);
  EXPECT_EQ(engine->unregisterCount, 1);  // teardown unwound the publish
}

// A worker with no peers configured is immediately ready after publishing.
TEST(MooncakeMigrationWorker, BringUpWithNoPeersIsReady) {
  auto engine = std::make_shared<FakeTransferEngine>();
  MigrationWorkerConfig cfg;
  cfg.host_dram_bytes = 4096;
  cfg.segment_name = "self";  // no peer_segment_names

  MooncakeMigrationWorker worker{cfg, engine, fastDiscoveryService(5)};
  EXPECT_TRUE(worker.bringUp());
  EXPECT_TRUE(worker.peers().empty());
  EXPECT_EQ(engine->registerCount, 1);
}

// The destructor tears down exactly once. The worker no longer owns a hold-
// loop (process lifetime is the binary's job), so the only teardown trigger is
// stack scope; this test guards the idempotency invariant on that path.
TEST(MooncakeMigrationWorker, DestructorTearsDownOnce) {
  auto engine = std::make_shared<FakeTransferEngine>();
  engine->resolveAfterMisses = {{"peer-0", 0}};
  MigrationWorkerConfig cfg;
  cfg.host_dram_bytes = 4096;
  cfg.segment_name = "self";
  cfg.peer_segment_names = {"peer-0"};

  {
    MooncakeMigrationWorker worker{cfg, engine, fastDiscoveryService(5)};
    ASSERT_TRUE(worker.bringUp());
    EXPECT_EQ(engine->unregisterCount, 0);
  }
  EXPECT_EQ(engine->unregisterCount, 1);
}

}  // namespace
}  // namespace tt::transport
