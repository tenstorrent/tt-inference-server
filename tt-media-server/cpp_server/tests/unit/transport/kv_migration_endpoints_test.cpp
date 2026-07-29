// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_endpoints.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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
        /*localTableBlob=*/std::vector<uint8_t>{}, K_RECV_TIMEOUT,
        K_POLL_INTERVAL);
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

TEST(KvControlChannelConnectorTest, OpenChannelAddsLatePeer) {
  std::unordered_map<std::string, Endpoint> eps{{"D0", {"10.0.0.1", 7001}}};

  KvControlChannelConnector connector(
      eps,
      [](const Endpoint& /*ep*/) -> std::shared_ptr<sockets::ISocketTransport> {
        return std::make_shared<BlockingFakeTransport>(
            std::make_shared<Pipe>(), std::make_shared<Pipe>());
      });

  ASSERT_TRUE(connector.openChannels());
  EXPECT_EQ(connector.channelCount(), 1u);

  EXPECT_TRUE(connector.openChannel("D1", Endpoint{"10.0.0.2", 7002}));
  EXPECT_EQ(connector.channelCount(), 2u);
  auto channels = connector.channels();
  ASSERT_TRUE(channels.count("D1"));
  EXPECT_NE(channels["D1"], nullptr);

  // Idempotent: second open of the same name is a no-op success.
  EXPECT_TRUE(connector.openChannel("D1", Endpoint{"10.0.0.2", 7002}));
  EXPECT_EQ(connector.channelCount(), 2u);
}

TEST(KvControlChannelConnectorTest, ReplaceChannelMovesEndpoint) {
  std::unordered_map<std::string, Endpoint> eps{{"D0", {"10.0.0.1", 7001}}};
  int factoryCalls = 0;

  KvControlChannelConnector connector(
      eps,
      [&](const Endpoint& /*ep*/)
          -> std::shared_ptr<sockets::ISocketTransport> {
        ++factoryCalls;
        return std::make_shared<BlockingFakeTransport>(
            std::make_shared<Pipe>(), std::make_shared<Pipe>());
      });

  ASSERT_TRUE(connector.openChannels());
  EXPECT_EQ(factoryCalls, 1);
  auto first = connector.channels().at("D0");

  // Same endpoint: no rebuild.
  EXPECT_TRUE(connector.replaceChannel("D0", Endpoint{"10.0.0.1", 7001}));
  EXPECT_EQ(factoryCalls, 1);
  EXPECT_EQ(connector.channels().at("D0"), first);

  // New host:port: tear down and dial again.
  EXPECT_TRUE(connector.replaceChannel("D0", Endpoint{"10.0.0.9", 7009}));
  EXPECT_EQ(factoryCalls, 2);
  EXPECT_NE(connector.channels().at("D0"), first);
  auto ep = connector.endpoint("D0");
  ASSERT_TRUE(ep.has_value());
  EXPECT_EQ(ep->host, "10.0.0.9");
  EXPECT_EQ(ep->port, 7009);
}

// Holding a channels() snapshot must keep the old channel alive across
// replaceChannel() — mirrors migrate() / mesh-watch concurrent with
// rediscovery.
TEST(KvControlChannelConnectorTest, ReplaceKeepsHeldChannelAlive) {
  std::unordered_map<std::string, Endpoint> eps{{"D0", {"10.0.0.1", 7001}}};
  int factoryCalls = 0;

  KvControlChannelConnector connector(
      eps,
      [&](const Endpoint& /*ep*/)
          -> std::shared_ptr<sockets::ISocketTransport> {
        ++factoryCalls;
        return std::make_shared<BlockingFakeTransport>(
            std::make_shared<Pipe>(), std::make_shared<Pipe>());
      });

  ASSERT_TRUE(connector.openChannels());
  auto held = connector.channels().at("D0");
  ASSERT_NE(held, nullptr);
  const KvControlChannel* raw = held.get();
  EXPECT_GE(held.use_count(), 2);  // connector + held

  ASSERT_TRUE(connector.replaceChannel("D0", Endpoint{"10.0.0.9", 7009}));
  EXPECT_EQ(factoryCalls, 2);
  // Connector dropped its ref; held keeps the old object alive.
  EXPECT_EQ(held.get(), raw);
  EXPECT_EQ(held.use_count(), 1);
  EXPECT_NE(connector.channels().at("D0"), held);
  EXPECT_NE(connector.channels().at("D0").get(), raw);
}

TEST(KvMigrationMultiHostSenderTest, AddHostWiresLatePeerForMigrate) {
  auto reg = std::make_shared<FakeRegistry>();
  auto senderEngine = std::make_shared<FakeTransferEngine>(reg, "P");
  auto prefill = std::make_shared<InMemoryKvTable>(
      buildTable("P", {1, 0}, {1, 1}, {1, 2}, {1, 3}, 0x1000, 0, 0x2000, 0));
  auto decode = std::make_shared<InMemoryKvTable>(buildTableSplitHosts(
      "D0", "D1", {2, 0}, {2, 1}, {3, 0}, {3, 1}, 0x8000, 0, 0x9000, 1));

  FakeDeviceIo prefillDev;
  seedPrefill(prefillDev);
  FakeDeviceIo devD0, devD1;
  DecodeHost d0(reg, decode, "D0", "segD0", devD0);
  DecodeHost d1(reg, decode, "D1", "segD1", devD1);

  std::unordered_map<std::string, Endpoint> eps{{"D0", {"127.0.0.1", 7001}}};
  std::unordered_map<uint16_t, std::shared_ptr<BlockingFakeTransport>> clients{
      {7001, d0.clientTp}, {7002, d1.clientTp}};
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
  // Without D1 the fan-out must fail (layer 1 lives on D1).
  EXPECT_FALSE(multiHost.migrate(0xAB, wholeSlot5()));

  ASSERT_TRUE(connector.openChannel("D1", Endpoint{"127.0.0.1", 7002}));
  auto channels = connector.channels();
  ASSERT_TRUE(multiHost.addHost("D1", channels.at("D1")));
  EXPECT_EQ(multiHost.hostCount(), 2u);
  EXPECT_TRUE(multiHost.migrate(0xAC, wholeSlot5()));

  d0.server->stop();
  d1.server->stop();
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

// Listen transport that supports multi-accept: each fireAccept() hands a new
// peer to the server's onAccept path (same as TcpSocketTransport).
class MultiAcceptListenFake : public sockets::ISocketTransport {
 public:
  bool initializeAsServer(uint16_t) override { return true; }
  bool initializeAsClient(const std::string&, uint16_t) override {
    return true;
  }
  void start() override {}
  void stop() override {}
  bool isConnected() const override { return true; }
  std::string getStatus() const override { return "multi-accept-fake"; }
  bool sendRawData(std::span<const uint8_t>) override { return false; }
  std::vector<uint8_t> receiveRawData() override { return {}; }
  void setConnectionLostCallback(std::function<void()>) override {}
  void setConnectionEstablishedCallback(std::function<void()>) override {}

  bool enableMultiAccept(AcceptHandler acceptHandler) override {
    handler = std::move(acceptHandler);
    return true;
  }

  void fireAccept(std::shared_ptr<sockets::ISocketTransport> peer) {
    ASSERT_TRUE(handler);
    handler(std::move(peer));
  }

 private:
  AcceptHandler handler;
};

// Peer that reports CLOSED immediately so KvMigrationReceiver::run() exits.
class ImmediatelyClosedPeer : public sockets::ISocketTransport {
 public:
  bool initializeAsServer(uint16_t) override { return true; }
  bool initializeAsClient(const std::string&, uint16_t) override {
    return true;
  }
  void start() override {}
  void stop() override {}
  bool isConnected() const override { return false; }
  std::string getStatus() const override { return "closed-peer"; }
  bool sendRawData(std::span<const uint8_t>) override { return false; }
  std::vector<uint8_t> receiveRawData() override { return {}; }
  sockets::ReceiveResult tryReceiveMessage() override {
    return {sockets::ReceiveStatus::CLOSED, {}};
  }
  void setConnectionLostCallback(std::function<void()>) override {}
  void setConnectionEstablishedCallback(std::function<void()>) override {}
};

// Finished sessions free themselves when run() exits (no wait for the next
// accept) so large peer-table blobs are not retained indefinitely.
TEST(KvMigrationReceiverServerTest, ReapsFinishedSessionsWithoutNextAccept) {
  auto reg = std::make_shared<FakeRegistry>();
  auto table = std::make_shared<InMemoryKvTable>(
      buildTable("D0", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  auto engine = std::make_shared<FakeTransferEngine>(reg, "segD0");
  FakeDeviceIo dev;
  MooncakeKvReceiver receiver(engine, dev, table, "D0", "segD0");

  auto listen = std::make_shared<MultiAcceptListenFake>();
  KvMigrationReceiverServer server(
      /*port=*/18650, [listen](uint16_t) { return listen; }, receiver,
      /*localTableBlob=*/{}, K_RECV_TIMEOUT, K_POLL_INTERVAL);
  ASSERT_TRUE(server.start());

  listen->fireAccept(std::make_shared<ImmediatelyClosedPeer>());
  for (int i = 0; i < 200 && server.activeSessionCount() > 0; ++i) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_EQ(server.activeSessionCount(), 0u);

  server.stop();
}

// B1: finished sessions must also be reaped on the next accept (belt+suspenders
// if a session's self-reap raced with try_lock failure under stop()).
TEST(KvMigrationReceiverServerTest, ReapsFinishedSessionsOnAccept) {
  auto reg = std::make_shared<FakeRegistry>();
  auto table = std::make_shared<InMemoryKvTable>(
      buildTable("D0", {2, 0}, {2, 1}, {2, 2}, {2, 3}, 0x8000, 0, 0x9000, 1));
  auto engine = std::make_shared<FakeTransferEngine>(reg, "segD0");
  FakeDeviceIo dev;
  MooncakeKvReceiver receiver(engine, dev, table, "D0", "segD0");

  auto listen = std::make_shared<MultiAcceptListenFake>();
  KvMigrationReceiverServer server(
      /*port=*/18650, [listen](uint16_t) { return listen; }, receiver,
      /*localTableBlob=*/{}, K_RECV_TIMEOUT, K_POLL_INTERVAL);
  ASSERT_TRUE(server.start());

  listen->fireAccept(std::make_shared<ImmediatelyClosedPeer>());
  // run() should observe CLOSED and finish quickly.
  for (int i = 0; i < 200 && server.activeSessionCount() > 0; ++i) {
    std::this_thread::sleep_for(1ms);
  }

  listen->fireAccept(std::make_shared<ImmediatelyClosedPeer>());
  for (int i = 0; i < 200 && server.activeSessionCount() > 0; ++i) {
    std::this_thread::sleep_for(1ms);
  }

  // After the second accept, the first finished session was reaped; at most
  // one (the latest, possibly still exiting) may remain briefly.
  EXPECT_LE(server.activeSessionCount(), 1u);

  server.stop();
  EXPECT_EQ(server.activeSessionCount(), 0u);
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
