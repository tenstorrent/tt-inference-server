// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "transport/host_dram_storage_backend.hpp"
#include "transport/i_storage_backend.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/transfer_types.hpp"

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

// The Transfer Engine composes a storage backend and reports its medium, and
// exposes that same backend through storage() for the bounce-buffer flow.
TEST(MooncakeTransferEngine, ComposesStorageBackend) {
  auto storage = std::make_shared<HostDramStorageBackend>();
  std::unique_ptr<ITransferEngine> engine =
      std::make_unique<MooncakeTransferEngine>(storage);
  ASSERT_NE(engine, nullptr);
  EXPECT_EQ(engine->storageMedium(), StorageMedium::HOST_DRAM);
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

}  // namespace
}  // namespace tt::transport
