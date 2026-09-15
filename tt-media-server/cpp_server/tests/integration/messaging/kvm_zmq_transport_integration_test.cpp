// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Integration test for the ZMQ transport + RemoteKVManagerZmqImpl.
//
// Wires up a real KvmZmqTransport (engine-side DEALER) and a mock
// kv_manager command ROUTER to exercise the direct on-wire framing.
//
// Uses tcp:// on 127.0.0.1 with a probed ephemeral port range rather than
// inproc:// because inproc requires a shared ZMQ context between the two
// sides. `KvmZmqTransport` owns its own context, so tcp:// is what proves
// the cross-context production wiring works.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zmq.hpp>

#include "messaging/kvm_command_message.hpp"
#include "messaging/kvm_zmq_transport.hpp"
#include "services/remote_kv_manager.hpp"
#include "services/remote_kv_manager_zmq_impl.hpp"

namespace tt {
namespace {

using namespace std::chrono_literals;
using tt::messaging::KvmResponseMessage;
using tt::messaging::KvmZmqTransport;
using tt::messaging::KvmZmqTransportConfig;
using tt::messaging::parseKvmCommand;
using tt::messaging::serialize;
using tt::services::MigrationRequest;
using tt::services::MigrationStatus;
using tt::services::RemoteKVManagerZmqImpl;

// Small deadline-based spin. Avoids fixed sleeps that either flake under
// load or slow down the suite unnecessarily.
template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

/** Mock of kv_manager's command ROUTER. */
class MockKvManagerPeer {
 public:
  MockKvManagerPeer(zmq::context_t& ctx,
                    MigrationStatus ackStatus = MigrationStatus::SUCCESSFUL)
      : router(ctx, zmq::socket_type::router), ackStatus(ackStatus) {
    router.set(zmq::sockopt::linger, 0);
    router.bind("tcp://127.0.0.1:*");
    endpoint = router.get(zmq::sockopt::last_endpoint);
  }

  void start() {
    running = true;
    thread = std::thread([this] { runLoop(); });
  }

  void stop() {
    running = false;
    if (thread.joinable()) {
      thread.join();
    }
  }

  ~MockKvManagerPeer() { stop(); }

  uint64_t receivedCount() const {
    return received.load(std::memory_order_relaxed);
  }

  const std::string& getEndpoint() const { return endpoint; }

 private:
  void runLoop() {
    while (running.load(std::memory_order_relaxed)) {
      zmq::pollitem_t items[] = {
          {static_cast<void*>(router), 0, ZMQ_POLLIN, 0}};
      const int rc = zmq::poll(items, 1, 10ms);
      if (rc <= 0 || (items[0].revents & ZMQ_POLLIN) == 0) continue;

      zmq::message_t route;
      auto routeResult = router.recv(route, zmq::recv_flags::none);
      if (!routeResult.has_value() || !route.more()) continue;
      zmq::message_t payloadMsg;
      auto payloadResult = router.recv(payloadMsg, zmq::recv_flags::none);
      if (!payloadResult.has_value() || payloadMsg.more()) continue;

      const std::string payload(static_cast<const char*>(payloadMsg.data()),
                                payloadMsg.size());
      auto cmd = parseKvmCommand(payload);
      if (!cmd.has_value()) continue;
      received.fetch_add(1, std::memory_order_relaxed);

      const std::string ack = serialize(KvmResponseMessage{
          .command_id = cmd->command_id,
          .migration_id = cmd->migration_id,
          .status = ackStatus,
      });

      zmq::message_t payloadOut(ack.data(), ack.size());
      (void)router.send(route, zmq::send_flags::sndmore);
      (void)router.send(payloadOut, zmq::send_flags::none);
    }
  }

  zmq::socket_t router;
  std::string endpoint;
  MigrationStatus ackStatus;
  std::atomic<bool> running{false};
  std::atomic<uint64_t> received{0};
  std::thread thread;
};

MigrationRequest makeRequest(uint32_t layerBegin, uint32_t layerEnd) {
  return MigrationRequest{
      .src_slot = 1,
      .dst_slot = 2,
      .layer_begin = layerBegin,
      .layer_end = layerEnd,
      .src_position_begin = 0,
      .src_position_end = 128,
      .dst_position_begin = 0,
      .dst_position_end = 128,
      .migration_id = 42,
  };
}

class KvmZmqTcpFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // We deliberately don't share `ctx` with KvmZmqTransport — the point is
    // to prove the two sides don't need to.
    ctx = std::make_unique<zmq::context_t>(/*io_threads=*/1);
  }

  void TearDown() override {
    if (peer) peer->stop();
    peer.reset();
    manager.reset();
    ctx.reset();
  }

  void connectTransport(
      MigrationStatus ackStatus = MigrationStatus::SUCCESSFUL) {
    peer = std::make_unique<MockKvManagerPeer>(*ctx, ackStatus);
    transport = std::make_unique<KvmZmqTransport>(
        KvmZmqTransportConfig{.endpoint = peer->getEndpoint()});
    peer->start();
  }

  void connectWithoutPeer() {
    zmq::socket_t probe(*ctx, zmq::socket_type::router);
    probe.set(zmq::sockopt::linger, 0);
    probe.bind("tcp://127.0.0.1:*");
    const std::string endpoint = probe.get(zmq::sockopt::last_endpoint);
    probe.close();
    transport = std::make_unique<KvmZmqTransport>(
        KvmZmqTransportConfig{.endpoint = endpoint});
  }

  std::unique_ptr<zmq::context_t> ctx;
  std::unique_ptr<KvmZmqTransport> transport;
  std::unique_ptr<RemoteKVManagerZmqImpl> manager;
  std::unique_ptr<MockKvManagerPeer> peer;
};

TEST_F(KvmZmqTcpFixture, SingleMigrateGetsAcked) {
  connectTransport();
  ASSERT_TRUE(transport);

  manager = std::make_unique<RemoteKVManagerZmqImpl>(
      std::move(transport), /*timeout=*/2s, /*sweep=*/50ms,
      /*drainPollMs=*/5);

  const uint64_t id = manager->migrate(makeRequest(0, 32));
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor([&] {
    return manager->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
  }));
  EXPECT_GE(peer->receivedCount(), 1u);
}

TEST_F(KvmZmqTcpFixture, PerLayerBurstIsFullyAcked) {
  connectTransport();
  ASSERT_TRUE(transport);

  manager = std::make_unique<RemoteKVManagerZmqImpl>(std::move(transport), 5s,
                                                     50ms, /*drainPollMs=*/5);

  // Simulate the 61-layer burst the PrefillScheduler will issue: one
  // migrate() call per layer, each with layer_begin == layer_end - 1.
  constexpr uint32_t numLayers = 61;
  std::vector<uint64_t> ids;
  ids.reserve(numLayers);
  for (uint32_t layer = 0; layer < numLayers; ++layer) {
    ids.push_back(manager->migrate(makeRequest(layer, layer + 1)));
  }

  ASSERT_TRUE(waitFor(
      [&] {
        for (uint64_t id : ids) {
          if (manager->getMigrationStatus(id) != MigrationStatus::SUCCESSFUL) {
            return false;
          }
        }
        return true;
      },
      3s));
  EXPECT_EQ(peer->receivedCount(), numLayers);
}

TEST_F(KvmZmqTcpFixture, FailedAckPropagates) {
  connectTransport(MigrationStatus::FAILED);
  ASSERT_TRUE(transport);

  manager = std::make_unique<RemoteKVManagerZmqImpl>(std::move(transport), 5s,
                                                     50ms, 5);

  const uint64_t id = manager->migrate(makeRequest(0, 1));
  ASSERT_TRUE(waitFor([&] {
    return manager->getMigrationStatus(id) == MigrationStatus::FAILED;
  }));
}

TEST_F(KvmZmqTcpFixture, TimeoutFiresWhenPeerNeverAcks) {
  connectWithoutPeer();
  ASSERT_TRUE(transport);
  // Deliberately no peer — no one echoes the ack. The sweeper should
  // eventually flip the migration to FAILED.
  manager = std::make_unique<RemoteKVManagerZmqImpl>(std::move(transport),
                                                     /*timeout=*/100ms,
                                                     /*sweep=*/10ms,
                                                     /*drainPollMs=*/5);

  const uint64_t id = manager->migrate(makeRequest(0, 1));
  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getMigrationStatus(id) == MigrationStatus::FAILED;
      },
      1s));
}

}  // namespace
}  // namespace tt
