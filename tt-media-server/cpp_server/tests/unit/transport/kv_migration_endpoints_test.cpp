// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_endpoints.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "sockets/i_socket_transport.hpp"
#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_migration_multi_host_sender.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport_test_fakes.hpp"

namespace tt::transport {
namespace {

using namespace std::chrono_literals;
using test::BlockingFakeTransport;
using test::buildTable;
using test::buildTableSplitHosts;
using test::FakeDeviceIo;
using test::FakeRegistry;
using test::FakeTransferEngine;
using test::K_CHUNK;
using test::makeContent;
using test::Pipe;
using test::wholeSlot5;

using Endpoint = KvControlChannelConnector::Endpoint;

// Fast channel timings so the not-ready/timeout paths never slow the suite.
constexpr auto K_RECV_TIMEOUT = 1000ms;
constexpr auto K_POLL_INTERVAL = 1ms;

void seedPrefill(FakeDeviceIo& dev) {
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    dev.put(encodeDevice({1, 0}), makeNocAddr(0, 0x1000 + idx * K_CHUNK),
            makeContent(0, p));
    dev.put(encodeDevice({1, 2}), makeNocAddr(0, 0x2000 + idx * K_CHUNK),
            makeContent(1, p));
  }
}

// One decode host stood up behind a KvMigrationReceiverServer over a loopback
// fake socket pair. clientTp is the matching sender-side end the connector
// wraps. The MooncakeKvReceiver registers its full-table mirror as the segment.
struct DecodeHost {
  std::shared_ptr<Pipe> ab{std::make_shared<Pipe>()};  // sender -> receiver
  std::shared_ptr<Pipe> ba{std::make_shared<Pipe>()};  // receiver -> sender
  std::shared_ptr<BlockingFakeTransport> clientTp;
  std::shared_ptr<FakeTransferEngine> engine;
  std::unique_ptr<MooncakeKvReceiver> receiver;
  std::unique_ptr<KvMigrationReceiverServer> server;

  DecodeHost(const std::shared_ptr<FakeRegistry>& reg,
             const std::shared_ptr<const IKvTable>& table,
             const std::string& host, const std::string& seg, IDeviceIo& dev) {
    clientTp = std::make_shared<BlockingFakeTransport>(/*in=*/ba, /*out=*/ab);
    engine = std::make_shared<FakeTransferEngine>(reg, seg);
    receiver =
        std::make_unique<MooncakeKvReceiver>(engine, dev, table, host, seg);
    auto serverTp =
        std::make_shared<BlockingFakeTransport>(/*in=*/ab, /*out=*/ba);
    server = std::make_unique<KvMigrationReceiverServer>(
        /*port=*/0, [serverTp](uint16_t) { return serverTp; }, *receiver,
        K_RECV_TIMEOUT, K_POLL_INTERVAL);
    server->start();
  }
  ~DecodeHost() {
    if (server) server->stop();
  }
};

// ---------------------------------------------------------------------------
// KvControlChannelConnector — channel construction & failure handling
// ---------------------------------------------------------------------------

TEST(KvControlChannelConnectorTest, OpensOneChannelPerEndpoint) {
  std::unordered_map<std::string, Endpoint> eps{{"D0", {"10.0.0.1", 7001}},
                                                {"D1", {"10.0.0.2", 7002}}};

  std::unordered_map<uint16_t, Endpoint> seen;
  KvControlChannelConnector connector(
      eps,
      [&](const Endpoint& ep) -> std::shared_ptr<sockets::ISocketTransport> {
        seen[ep.port] = ep;
        return std::make_shared<BlockingFakeTransport>(
            std::make_shared<Pipe>(), std::make_shared<Pipe>());
      });

  EXPECT_TRUE(connector.openChannels());
  auto channels = connector.channels();
  EXPECT_EQ(channels.size(), 2u);
  ASSERT_TRUE(channels.count("D0"));
  ASSERT_TRUE(channels.count("D1"));
  EXPECT_NE(channels["D0"], nullptr);
  // The factory saw each host's real endpoint (host + port), not the logical
  // key.
  EXPECT_EQ(seen[7001].host, "10.0.0.1");
  EXPECT_EQ(seen[7002].host, "10.0.0.2");
}

TEST(KvControlChannelConnectorTest, SkipsHostWhenFactoryFailsButKeepsRest) {
  std::unordered_map<std::string, Endpoint> eps{{"D0", {"10.0.0.1", 7001}},
                                                {"D1", {"10.0.0.2", 7002}}};

  KvControlChannelConnector connector(
      eps,
      [](const Endpoint& ep) -> std::shared_ptr<sockets::ISocketTransport> {
        if (ep.port == 7002) return nullptr;  // D1 unreachable
        return std::make_shared<BlockingFakeTransport>(
            std::make_shared<Pipe>(), std::make_shared<Pipe>());
      });

  EXPECT_FALSE(connector.openChannels());  // not every endpoint got a transport
  auto channels = connector.channels();
  EXPECT_EQ(channels.size(), 1u);
  EXPECT_TRUE(channels.count("D0"));
  EXPECT_FALSE(channels.count("D1"));
}

// ---------------------------------------------------------------------------
// KvMigrationReceiverServer — start failure
// ---------------------------------------------------------------------------

TEST(KvMigrationReceiverServerTest, StartFailsWhenTransportFactoryReturnsNull) {
  auto reg = std::make_shared<FakeRegistry>();
  auto table = std::make_shared<InMemoryKvTable>(
      buildTable("D0", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  auto engine = std::make_shared<FakeTransferEngine>(reg, "segD0");
  FakeDeviceIo dev;
  MooncakeKvReceiver receiver(engine, dev, table, "D0", "segD0");

  KvMigrationReceiverServer server(
      /*port=*/0, [](uint16_t) { return nullptr; }, receiver);
  EXPECT_FALSE(server.start());
  EXPECT_FALSE(server.running());
}

// ---------------------------------------------------------------------------
// End to end: connector + receiver server drive the real data plane
// ---------------------------------------------------------------------------

TEST(KvMigrationEndpoints, SingleHostMigratesThroughConnectorAndServer) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  // Single decode host "D0" owns both layers.
  auto decode = std::make_shared<InMemoryKvTable>(
      buildTable("D0", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0;
  DecodeHost d0(reg, decode, "D0", "segD0", devD0);

  std::unordered_map<std::string, Endpoint> eps{{"D0", {"127.0.0.1", 7001}}};
  std::unordered_map<uint16_t, std::shared_ptr<BlockingFakeTransport>> clients{
      {7001, d0.clientTp}};
  KvControlChannelConnector connector(
      eps,
      [&](const Endpoint& ep) -> std::shared_ptr<sockets::ISocketTransport> {
        return clients.at(ep.port);
      },
      K_RECV_TIMEOUT, K_POLL_INTERVAL);
  ASSERT_TRUE(connector.openChannels());

  KvMigrationMultiHostSender multiHost(senderEngine, prefillDev, prefill,
                                       decode, "P", connector.channels());
  EXPECT_EQ(multiHost.hostCount(), 1u);
  EXPECT_TRUE(multiHost.migrate(0x77, wholeSlot5()));

  d0.server->stop();

  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    EXPECT_EQ(
        devD0.get(encodeDevice({2, 0}), makeNocAddr(0, 0x8000 + idx * K_CHUNK)),
        makeContent(0, p));
    EXPECT_EQ(
        devD0.get(encodeDevice({2, 2}), makeNocAddr(1, 0x9000 + idx * K_CHUNK)),
        makeContent(1, p));
  }
}

TEST(KvMigrationEndpoints, TwoHostFanOutThroughConnectorAndServers) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  // layer 0 -> D0, layer 1 -> D1.
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0, devD1;
  DecodeHost d0(reg, decode, "D0", "segD0", devD0);
  DecodeHost d1(reg, decode, "D1", "segD1", devD1);

  std::unordered_map<std::string, Endpoint> eps{{"D0", {"127.0.0.1", 7001}},
                                                {"D1", {"127.0.0.1", 7002}}};
  std::unordered_map<uint16_t, std::shared_ptr<BlockingFakeTransport>> clients{
      {7001, d0.clientTp}, {7002, d1.clientTp}};
  KvControlChannelConnector connector(
      eps,
      [&](const Endpoint& ep) -> std::shared_ptr<sockets::ISocketTransport> {
        return clients.at(ep.port);
      },
      K_RECV_TIMEOUT, K_POLL_INTERVAL);
  ASSERT_TRUE(connector.openChannels());
  EXPECT_EQ(connector.channelCount(), 2u);

  KvMigrationMultiHostSender multiHost(senderEngine, prefillDev, prefill,
                                       decode, "P", connector.channels());
  EXPECT_EQ(multiHost.hostCount(), 2u);
  EXPECT_TRUE(multiHost.migrate(0x99, wholeSlot5()));

  d0.server->stop();
  d1.server->stop();

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
