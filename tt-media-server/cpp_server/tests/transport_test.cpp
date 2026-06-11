// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "transport/device_dram_storage_backend.hpp"
#include "transport/host_dram_storage_backend.hpp"
#include "transport/i_storage_backend.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/mooncake_migration_worker.hpp"
#include "transport/mooncake_transfer_engine.hpp"
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
  MooncakeMigrationWorker sender(senderCfg, engine);

  const std::vector<uint8_t> tensor(deviceRegion.size(), 0xAB);
  EXPECT_TRUE(sender.writeTensorOnSender(tensor));
  EXPECT_EQ(deviceRegion, tensor);  // tensor landed in "device DRAM"

  MigrationWorkerConfig receiverCfg = senderCfg;
  receiverCfg.role = MigrationRole::Receiver;
  MooncakeMigrationWorker receiver(receiverCfg, engine);
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

  MooncakeMigrationWorker sender(config, engine);
  const std::vector<uint8_t> tensor(config.tensor_bytes, 0xAB);
  // Sender cannot run the receiver step.
  EXPECT_FALSE(sender.verifyTensorOnReceiver(tensor));
  // The transport hop has no init'd engine / peer segment here.
  EXPECT_FALSE(sender.transferToReceiver());

  config.role = MigrationRole::Receiver;
  MooncakeMigrationWorker receiver(config, engine);
  // Receiver cannot run the sender steps.
  EXPECT_FALSE(receiver.writeTensorOnSender(tensor));
  EXPECT_FALSE(receiver.transferToReceiver());
}

}  // namespace
}  // namespace tt::transport
