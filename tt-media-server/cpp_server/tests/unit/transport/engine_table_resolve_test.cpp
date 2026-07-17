// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/engine_table_resolve.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "sockets/tcp_socket_transport.hpp"
#include "transport/device_map.hpp"
#include "transport/device_map_io.hpp"
#include "transport/engine_table_handoff.hpp"
#include "transport/kv_chunk_address_table_adapter.hpp"
#include "transport_test_fakes.hpp"

#ifndef KV_TABLE_DECODE_PB_DEFAULT
#define KV_TABLE_DECODE_PB_DEFAULT ""
#endif

namespace tt::transport {
namespace {

DeviceMap makeDeviceMap() {
  DeviceMap dm;
  dm.set(FabricNode{2, 0}, 0xAAAA000000000001ull);
  dm.set(FabricNode{3, 1}, 0xBBBB000000000002ull);
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

void writeTempDevMap(const std::string& path, const DeviceMap& dm) {
  std::ofstream out(path);
  for (const auto& [device, umd] : dm.entries()) {
    out << (device >> 16) << ' ' << (device & 0xFFFFu) << ' ' << umd << '\n';
  }
}

TEST(DeviceMapIo, ParsesMeshChipUmdLines) {
  std::istringstream input("1 2 100\n3 4 200\n");
  const DeviceMap dm = loadDeviceMapStream(input);
  EXPECT_EQ(dm.size(), 2u);
  EXPECT_EQ(dm.umdChip(FabricNode{1, 2}), 100u);
  EXPECT_EQ(dm.umdChip(FabricNode{3, 4}), 200u);
}

TEST(EngineTableResolve, FileModeLoadsTableBlobAndDeviceMap) {
  if (!KvChunkAddressTableAdapter::available()) {
    GTEST_SKIP() << "ENABLE_KV_TABLE is OFF";
  }
  const std::string tablePath =
      envOr("KV_TABLE_DECODE_PB", KV_TABLE_DECODE_PB_DEFAULT);
  if (!readable(tablePath)) {
    GTEST_SKIP() << "missing decode .pb (set KV_TABLE_DECODE_PB)";
  }

  const std::string mapPath = "/tmp/engine_table_resolve_test.devmap";
  writeTempDevMap(mapPath, makeDeviceMap());

  auto resolved = resolveEngineTablesFromFiles(tablePath, mapPath);
  ASSERT_TRUE(resolved.has_value());
  ASSERT_NE(resolved->table, nullptr);
  EXPECT_EQ(resolved->blob, readFile(tablePath));
  EXPECT_EQ(resolved->deviceMap.size(), 2u);
  EXPECT_EQ(resolved->deviceMap.umdChip(FabricNode{2, 0}),
            0xAAAA000000000001ull);
}

TEST(EngineTableResolve, TableFromFileDeviceMapFromSocket) {
  if (!KvChunkAddressTableAdapter::available()) {
    GTEST_SKIP() << "ENABLE_KV_TABLE is OFF";
  }
  const std::string tablePath =
      envOr("KV_TABLE_DECODE_PB", KV_TABLE_DECODE_PB_DEFAULT);
  if (!readable(tablePath)) {
    GTEST_SKIP() << "missing decode .pb (set KV_TABLE_DECODE_PB)";
  }

  const DeviceMap deviceMap = makeDeviceMap();
  std::atomic<bool> stop{false};

  constexpr uint16_t kPort = 18777;
  ListenTransportFactory factory =
      [](uint16_t port) -> std::shared_ptr<sockets::ISocketTransport> {
    auto t = std::make_shared<sockets::TcpSocketTransport>();
    if (!t->initializeAsServer(port)) return nullptr;
    return t;
  };

  std::optional<ResolvedEngineTables> resolved;
  std::thread listener([&] {
    resolved = resolveEngineTables(kPort, factory, tablePath, /*deviceMapPath=*/"",
                                   stop);
  });

  std::shared_ptr<sockets::TcpSocketTransport> client;
  for (int attempt = 0; attempt < 50; ++attempt) {
    client = std::make_shared<sockets::TcpSocketTransport>();
    if (client->initializeAsClient("127.0.0.1", kPort)) {
      break;
    }
    client.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!client) {
    stop.store(true);
    listener.join();
    GTEST_SKIP() << "could not bind/connect handoff port " << kPort;
  }
  client->start();
  ASSERT_TRUE(sendEngineHandoff(*client, deviceMap));
  client->stop();

  listener.join();
  ASSERT_TRUE(resolved.has_value());
  ASSERT_NE(resolved->table, nullptr);
  EXPECT_EQ(resolved->blob, readFile(tablePath));
  EXPECT_EQ(resolved->deviceMap.umdChip(FabricNode{2, 0}),
            0xAAAA000000000001ull);
  EXPECT_EQ(resolved->deviceMap.size(), deviceMap.size());
}

TEST(EngineTableResolve, PeerPollSeesSendThenClose) {
  auto link = std::make_shared<test::Pipe>();
  test::BlockingFakeTransport producer(std::make_shared<test::Pipe>(), link);
  auto consumer = std::make_shared<test::BlockingFakeTransport>(
      link, std::make_shared<test::Pipe>());

  ASSERT_TRUE(sendEngineHandoff(producer, makeDeviceMap()));
  test::closePipe(link);

  std::atomic<bool> stop{false};
  auto deviceMap = awaitEngineHandoffOnPeer(*consumer, stop);
  ASSERT_TRUE(deviceMap.has_value());
  EXPECT_EQ(deviceMap->size(), 2u);
  EXPECT_EQ(deviceMap->umdChip(FabricNode{2, 0}), 0xAAAA000000000001ull);
}

}  // namespace
}  // namespace tt::transport
