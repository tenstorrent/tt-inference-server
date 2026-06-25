// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "transport/device_dram_storage_backend.hpp"
#include "transport/host_dram_storage_backend.hpp"
#include "transport/i_storage_backend.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/mooncake_migration_worker.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/peer_discovery_service.hpp"
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
  EXPECT_EQ(host->medium(), StorageMedium::HostDram);

  auto device = std::make_unique<DeviceDramStorageBackend>(
      std::make_shared<UmdDeviceAccess>(/*device_id=*/0));
  EXPECT_EQ(device->medium(), StorageMedium::DeviceDram);
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
  EXPECT_EQ(engine->storageMedium(), StorageMedium::DeviceDram);
  EXPECT_EQ(engine->storage(), storage);
}

#ifndef TT_TRANSPORT_WITH_MOONCAKE
// Without the Mooncake backend in the build, the transport methods report
// failure (not a live engine) without crashing. With Mooncake compiled in,
// init() touches the network/metadata service, so that path is exercised by
// the integration tests instead.
TEST(MooncakeTransferEngine, MethodsReportFailureWithoutMooncake) {
  MooncakeTransferEngine engine(std::make_shared<HostDramStorageBackend>());
  EXPECT_FALSE(engine.init(EngineConfig{}));

  std::vector<uint8_t> buffer(64, 0);
  EXPECT_FALSE(engine.registerLocalMemory(buffer.data(), buffer.size()));
  EXPECT_EQ(engine.openSegment("peer"), kInvalidSegment);

  TransferRequest request{TransferOp::Write, buffer.data(), kInvalidSegment, 0,
                          buffer.size()};
  EXPECT_EQ(engine.submitAndWait(request).state, TransferState::Failed);
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
  senderCfg.role = MigrationRole::Sender;
  senderCfg.peer_segment_name = "receiver";
  senderCfg.device_addr = addr;
  senderCfg.tensor_bytes = deviceRegion.size();
  MooncakeMigrationWorker sender(senderCfg, engine, nullptr);

  const std::vector<uint8_t> tensor(deviceRegion.size(), 0xAB);
  EXPECT_TRUE(sender.writeTensorOnSender(tensor));
  EXPECT_EQ(deviceRegion, tensor);  // tensor landed in "device DRAM"

  MigrationWorkerConfig receiverCfg = senderCfg;
  receiverCfg.role = MigrationRole::Receiver;
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
  config.role = MigrationRole::Sender;
  config.peer_segment_name = "receiver";
  config.device_addr = reinterpret_cast<uint64_t>(deviceRegion.data());
  config.tensor_bytes = deviceRegion.size();

  MooncakeMigrationWorker sender(config, engine, nullptr);
  const std::vector<uint8_t> tensor(config.tensor_bytes, 0xAB);
  // Sender cannot run the receiver step.
  EXPECT_FALSE(sender.verifyTensorOnReceiver(tensor));
  // The transport hop has no init'd engine / peer segment here.
  EXPECT_FALSE(sender.transferToReceiver());

  config.role = MigrationRole::Receiver;
  MooncakeMigrationWorker receiver(config, engine, nullptr);
  // Receiver cannot run the sender steps.
  EXPECT_FALSE(receiver.writeTensorOnSender(tensor));
  EXPECT_FALSE(receiver.transferToReceiver());
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
    return StorageMedium::HostDram;
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
    if (it == resolveAfterMisses.end()) return kInvalidSegment;
    return attempt > it->second ? static_cast<SegmentHandle>(attempt)
                                : kInvalidSegment;
  }
  TransferStatus submitAndWait(const TransferRequest&) override {
    return {TransferState::Completed, 0};
  }
};

// Fast tunables so the polling path runs in milliseconds, not seconds.
PeerDiscoveryConfig fastDiscovery(int timeoutSec) {
  return PeerDiscoveryConfig{/*poll_interval_ms=*/1,
                             /*timeout_sec=*/timeoutSec};
}

// A fast PeerDiscoveryService to inject into the worker under test.
std::shared_ptr<PeerDiscoveryService> fastDiscoveryService(int timeoutSec) {
  return std::make_shared<PeerDiscoveryService>(fastDiscovery(timeoutSec));
}

// All peers already registered: a single sweep resolves every name.
TEST(PeerDiscoveryService, ResolvesAllPeersInOneSweep) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"a", 0}, {"b", 0}, {"c", 0}};
  PeerDiscoveryService discovery(fastDiscovery(5));

  const auto resolved = discovery.discover(engine, {"a", "b", "c"});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->size(), 3u);
  for (const auto& name : {"a", "b", "c"}) {
    EXPECT_EQ(engine.openAttempts[name], 1);  // resolved first try
    EXPECT_NE(resolved->at(name), kInvalidSegment);
  }
}

// A peer that isn't registered yet is retried; only the unresolved name is
// re-polled, and discovery succeeds once it appears.
TEST(PeerDiscoveryService, RetriesOnlyUnresolvedUntilTheyAppear) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"ready", 0}, {"late", 2}};
  PeerDiscoveryService discovery(fastDiscovery(5));

  const auto resolved = discovery.discover(engine, {"ready", "late"});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->size(), 2u);
  EXPECT_EQ(engine.openAttempts["ready"], 1);  // resolved once, not re-polled
  EXPECT_GE(engine.openAttempts["late"], 3);   // 2 misses then a hit
}

// A peer that never registers makes discovery give up and return nullopt.
TEST(PeerDiscoveryService, TimesOutWhenAPeerNeverResolves) {
  FakeTransferEngine engine;
  engine.resolveAfterMisses = {{"present", 0}};  // "ghost" absent => never
  PeerDiscoveryService discovery(fastDiscovery(1));

  const auto resolved = discovery.discover(engine, {"present", "ghost"});
  EXPECT_FALSE(resolved.has_value());
  EXPECT_GE(engine.openAttempts["ghost"], 1);  // it did try
}

// No peers configured: discover() short-circuits to an empty (success) map
// without ever touching the engine.
TEST(PeerDiscoveryService, NoPeersResolvesToEmptyMap) {
  FakeTransferEngine engine;
  PeerDiscoveryService discovery(fastDiscovery(5));

  const auto resolved = discovery.discover(engine, {});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_TRUE(resolved->empty());
  EXPECT_TRUE(engine.callLog.empty());  // no openSegment attempts
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

  MooncakeMigrationWorker worker(cfg, engine, fastDiscoveryService(5));
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
  MooncakeMigrationWorker worker(zeroPool, engine, fastDiscoveryService(5));
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

  MooncakeMigrationWorker worker(cfg, engine, fastDiscoveryService(0));
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

  MooncakeMigrationWorker worker(cfg, engine, fastDiscoveryService(5));
  EXPECT_TRUE(worker.bringUp());
  EXPECT_TRUE(worker.peers().empty());
  EXPECT_EQ(engine->registerCount, 1);
}

// run() returns once stop is already requested and tears down exactly once
// (idempotent: the destructor's teardown must not unregister again).
TEST(MooncakeMigrationWorker, RunStopsAndTearsDownOnce) {
  auto engine = std::make_shared<FakeTransferEngine>();
  engine->resolveAfterMisses = {{"peer-0", 0}};
  MigrationWorkerConfig cfg;
  cfg.host_dram_bytes = 4096;
  cfg.segment_name = "self";
  cfg.peer_segment_names = {"peer-0"};

  {
    MooncakeMigrationWorker worker(cfg, engine, fastDiscoveryService(5));
    ASSERT_TRUE(worker.bringUp());
    std::atomic<bool> stop{true};  // already requested -> returns immediately
    worker.run(stop);
    EXPECT_EQ(engine->unregisterCount, 1);
  }  // destructor runs teardown again; must stay idempotent
  EXPECT_EQ(engine->unregisterCount, 1);
}

}  // namespace
}  // namespace tt::transport
