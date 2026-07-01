// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/engine_table_handoff.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "transport/device_map.hpp"
#include "transport/kv_chunk_address_table_adapter.hpp"
#include "transport/kv_table_provisioning.hpp"
#include "transport_test_fakes.hpp"

#ifndef KV_TABLE_DECODE_PB_DEFAULT
#define KV_TABLE_DECODE_PB_DEFAULT ""
#endif

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

std::string envOr(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string{v} : std::string{fallback};
}

bool readable(const std::string& path) {
  if (path.empty()) return false;
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  const auto size = f.tellg();
  std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

// ---------------------------------------------------------------------------
// Serialization — runs in every build
// ---------------------------------------------------------------------------

TEST(EngineTableHandoff, SerializeRoundTripsTableBlobAndDeviceMap) {
  const std::vector<uint8_t> tableBlob{0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x42};
  const DeviceMap dm = makeDeviceMap();

  const auto wire = serializeEngineHandoff(tableBlob, dm);
  const auto parsed = parseEngineHandoff(wire);
  ASSERT_TRUE(parsed.has_value());

  EXPECT_EQ(parsed->table_blob, tableBlob);
  EXPECT_EQ(parsed->device_map.size(), dm.size());
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{2, 0}), 0xAAAA000000000001ull);
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{2, 1}), 0xAAAA000000000002ull);
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{3, 0}), 0xBBBB000000000003ull);
  EXPECT_FALSE(parsed->device_map.umdChip(FabricNode{9, 9}).has_value());
}

TEST(EngineTableHandoff, SerializeRoundTripsEmptyDeviceMap) {
  const std::vector<uint8_t> tableBlob{1, 2, 3};
  const auto parsed = parseEngineHandoff(serializeEngineHandoff(tableBlob, {}));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->table_blob, tableBlob);
  EXPECT_TRUE(parsed->device_map.empty());
}

TEST(EngineTableHandoff, ParseRejectsTruncated) {
  const auto wire = serializeEngineHandoff({1, 2, 3, 4}, makeDeviceMap());
  for (std::size_t cut : {std::size_t{0}, wire.size() / 2, wire.size() - 1}) {
    std::span<const uint8_t> truncated(wire.data(), cut);
    EXPECT_FALSE(parseEngineHandoff(truncated).has_value())
        << "should reject truncation at " << cut;
  }
}

TEST(EngineTableHandoff, WireSendReceivePreservesPayload) {
  // One-directional: producer (engine) sends, consumer (worker) reads.
  auto link = std::make_shared<Pipe>();
  BlockingFakeTransport producer(std::make_shared<Pipe>(), /*out=*/link);
  BlockingFakeTransport consumer(/*in=*/link, std::make_shared<Pipe>());

  const std::vector<uint8_t> tableBlob{5, 6, 7, 8, 9};
  ASSERT_TRUE(sendEngineHandoff(producer, tableBlob, makeDeviceMap()));

  const auto raw = consumer.receiveRawData();
  const auto parsed = parseEngineHandoff(raw);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->table_blob, tableBlob);
  EXPECT_EQ(parsed->device_map.umdChip(FabricNode{3, 0}), 0xBBBB000000000003ull);
}

// ---------------------------------------------------------------------------
// Real table through the socket source — gated on ENABLE_KV_TABLE + .pb present
// ---------------------------------------------------------------------------

TEST(EngineTableHandoffRealTable, EngineHandsRealTableAndDeviceMapToWorker) {
  if (!KvChunkAddressTableAdapter::available()) {
    GTEST_SKIP() << "ENABLE_KV_TABLE is OFF; real-table handoff not built";
  }
  const std::string decodePath =
      envOr("KV_TABLE_DECODE_PB", KV_TABLE_DECODE_PB_DEFAULT);
  if (!readable(decodePath)) {
    GTEST_SKIP() << "missing decode .pb (set KV_TABLE_DECODE_PB)";
  }

  // The engine side: it already holds the device-built table (here we read the
  // bytes the model runner emitted) and resolves the device map from live chips.
  const std::vector<uint8_t> tableBlob = readFile(decodePath);
  const DeviceMap engineDeviceMap = makeDeviceMap();

  auto link = std::make_shared<Pipe>();
  auto producerTp =
      std::make_shared<BlockingFakeTransport>(std::make_shared<Pipe>(), link);
  auto consumerTp =
      std::make_shared<BlockingFakeTransport>(link, std::make_shared<Pipe>());

  ASSERT_TRUE(sendEngineHandoff(*producerTp, tableBlob, engineDeviceMap));

  // The worker side: pull table + device map through the seam.
  SocketEngineTableSource source(consumerTp);
  auto tables = source.fetch();
  ASSERT_TRUE(tables.has_value());
  ASSERT_NE(tables->table, nullptr);

  // The wire-delivered table addresses identically to the directly-loaded one.
  auto direct = deserializeKvTable(tableBlob);
  ASSERT_NE(direct, nullptr);
  auto a = direct->lookup(0, 0, 0);
  auto b = tables->table->lookup(0, 0, 0);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(b->noc_addr, a->noc_addr);
  EXPECT_EQ(b->size_bytes, a->size_bytes);

  // And the device map survived, so the worker can resolve a chip.
  EXPECT_EQ(tables->device_map.umdChip(FabricNode{2, 0}),
            0xAAAA000000000001ull);
  EXPECT_EQ(tables->device_map.size(), engineDeviceMap.size());
}

}  // namespace
}  // namespace tt::transport
