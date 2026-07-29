// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/engine_table_handoff.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "transport/device_map.hpp"
#include "transport_test_fakes.hpp"

namespace tt::transport {
namespace {

using test::BlockingFakeTransport;
using test::Pipe;

DeviceMap makeDeviceMap() {
  DeviceMap dm;
  dm.set(FabricNode{2, 0}, 0xAAAA000000000001ull);
  dm.set(FabricNode{2, 1}, 0xAAAA000000000002ull);
  dm.set(FabricNode{3, 0}, 0xBBBB000000000003ull);
  return dm;
}

TEST(EngineTableHandoff, SerializeRoundTripsDeviceMap) {
  const DeviceMap dm = makeDeviceMap();

  const auto wire = serializeEngineHandoff(dm);
  const auto parsed = parseEngineHandoff(wire);
  ASSERT_TRUE(parsed.has_value());

  EXPECT_EQ(parsed->device_map.size(), dm.size());
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{2, 0}),
            0xAAAA000000000001ull);
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{2, 1}),
            0xAAAA000000000002ull);
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{3, 0}),
            0xBBBB000000000003ull);
  EXPECT_FALSE(parsed->device_map.umdChip(FabricNode{9, 9}).has_value());
}

TEST(EngineTableHandoff, SerializeRoundTripsEmptyDeviceMap) {
  const auto parsed = parseEngineHandoff(serializeEngineHandoff({}));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_TRUE(parsed->device_map.empty());
}

TEST(EngineTableHandoff, ParseRejectsTruncated) {
  const auto wire = serializeEngineHandoff(makeDeviceMap());
  for (std::size_t cut : {std::size_t{0}, wire.size() / 2, wire.size() - 1}) {
    std::span<const uint8_t> truncated(wire.data(), cut);
    EXPECT_FALSE(parseEngineHandoff(truncated).has_value())
        << "should reject truncation at " << cut;
  }
}

TEST(EngineTableHandoff, WireSendReceivePreservesPayload) {
  auto link = std::make_shared<Pipe>();
  BlockingFakeTransport producer(std::make_shared<Pipe>(), /*out=*/link);
  BlockingFakeTransport consumer(/*in=*/link, std::make_shared<Pipe>());

  ASSERT_TRUE(sendEngineHandoff(producer, makeDeviceMap()));

  const auto raw = consumer.receiveRawData();
  const auto parsed = parseEngineHandoff(raw);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{3, 0}),
            0xBBBB000000000003ull);
}

TEST(EngineTableHandoff, SocketSourceFetchesDeviceMap) {
  auto link = std::make_shared<Pipe>();
  auto producerTp =
      std::make_shared<BlockingFakeTransport>(std::make_shared<Pipe>(), link);
  auto consumerTp =
      std::make_shared<BlockingFakeTransport>(link, std::make_shared<Pipe>());

  const DeviceMap engineDeviceMap = makeDeviceMap();
  ASSERT_TRUE(sendEngineHandoff(*producerTp, engineDeviceMap));

  SocketEngineDeviceMapSource source(consumerTp);
  auto deviceMap = source.fetch();
  ASSERT_TRUE(deviceMap.has_value());
  EXPECT_EQ(deviceMap->umdChip(FabricNode{2, 0}), 0xAAAA000000000001ull);
  EXPECT_EQ(deviceMap->size(), engineDeviceMap.size());
}

}  // namespace
}  // namespace tt::transport
