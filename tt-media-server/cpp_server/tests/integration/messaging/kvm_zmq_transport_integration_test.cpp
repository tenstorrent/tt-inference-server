// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Integration test for the ZMQ transport + RemoteKVManagerZmqImpl.
//
// Wires up a real KvmZmqTransport (engine side, PUB+SUB bind on loopback
// tcp:// URIs) and a mock kv_manager peer (SUB+PUB connect, mirroring
// `kvm::cp::command::ZmqCommandTransport`) to exercise the on-wire
// framing end-to-end without touching Kafka.
//
// Uses tcp:// on 127.0.0.1 with a probed ephemeral port range rather than
// inproc:// because inproc requires a shared ZMQ context between the two
// sides. `KvmZmqTransport` owns its own context, so tcp:// is what proves
// the cross-context production wiring works.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
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

constexpr const char* KVM_TOPIC = "L1";

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

/**
 * Mock kv_manager peer. Uses cppzmq to `zmq_connect` a SUB to the engine's
 * cmd endpoint and a PUB to the engine's reply endpoint — same shape as
 * kv_manager's real `ZmqCommandTransport::open` (see
 * `tt-d-gen/kv_manager/src/control_plane/command/zmq_command_transport.cpp`).
 * Every command it receives triggers a scripted ack (default SUCCESSFUL)
 * with the same command_id + migration_id.
 */
class MockKvManagerPeer {
 public:
  MockKvManagerPeer(const std::string& cmdEndpoint,
                    const std::string& replyEndpoint, zmq::context_t& ctx,
                    MigrationStatus ackStatus = MigrationStatus::SUCCESSFUL)
      : sub(ctx, zmq::socket_type::sub),
        pub(ctx, zmq::socket_type::pub),
        ackStatus(ackStatus) {
    sub.set(zmq::sockopt::linger, 0);
    pub.set(zmq::sockopt::linger, 0);
    sub.set(zmq::sockopt::subscribe, KVM_TOPIC);
    sub.connect(cmdEndpoint);
    pub.connect(replyEndpoint);
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

 private:
  void runLoop() {
    while (running.load(std::memory_order_relaxed)) {
      zmq::pollitem_t items[] = {{static_cast<void*>(sub), 0, ZMQ_POLLIN, 0}};
      const int rc = zmq::poll(items, 1, 10ms);
      if (rc <= 0 || (items[0].revents & ZMQ_POLLIN) == 0) continue;

      zmq::message_t topicMsg;
      auto tr = sub.recv(topicMsg, zmq::recv_flags::none);
      if (!tr.has_value()) continue;
      if (!topicMsg.more()) continue;
      zmq::message_t payloadMsg;
      auto pr = sub.recv(payloadMsg, zmq::recv_flags::none);
      if (!pr.has_value()) continue;

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

      zmq::message_t topicOut(KVM_TOPIC, std::strlen(KVM_TOPIC));
      zmq::message_t payloadOut(ack.data(), ack.size());
      (void)pub.send(topicOut, zmq::send_flags::sndmore);
      (void)pub.send(payloadOut, zmq::send_flags::none);
    }
  }

  zmq::socket_t sub;
  zmq::socket_t pub;
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

  KvmZmqTransportConfig endpoints(int port) const {
    return KvmZmqTransportConfig{
        .cmdEndpoint = "tcp://127.0.0.1:" + std::to_string(port),
        .replyEndpoint = "tcp://127.0.0.1:" + std::to_string(port + 1),
        .topic = KVM_TOPIC,
    };
  }

  // Bind a KvmZmqTransport on ephemeral port range, retrying on collision.
  // Returns the config used so the peer can connect to the same endpoints.
  KvmZmqTransportConfig bindTransport() {
    for (int port = 25610; port < 25650; port += 2) {
      auto cfg = endpoints(port);
      try {
        transport = std::make_unique<KvmZmqTransport>(cfg);
        return cfg;
      } catch (const std::exception&) {
        continue;
      }
    }
    ADD_FAILURE() << "failed to bind KvmZmqTransport on any test port";
    return endpoints(0);
  }

  std::unique_ptr<zmq::context_t> ctx;
  std::unique_ptr<KvmZmqTransport> transport;
  std::unique_ptr<RemoteKVManagerZmqImpl> manager;
  std::unique_ptr<MockKvManagerPeer> peer;
};

TEST_F(KvmZmqTcpFixture, SingleMigrateGetsAcked) {
  auto cfg = bindTransport();
  ASSERT_TRUE(transport);

  peer = std::make_unique<MockKvManagerPeer>(cfg.cmdEndpoint, cfg.replyEndpoint,
                                             *ctx);
  peer->start();

  manager = std::make_unique<RemoteKVManagerZmqImpl>(
      std::move(transport), /*timeout=*/2s, /*sweep=*/50ms,
      /*drainPollMs=*/5);

  // PUB/SUB slow-joiner: kv_manager's SUB has to finish connecting before
  // the engine's PUB emits or the first message is silently dropped. In
  // production this is mitigated by the sweeper (FAILED after `timeout`);
  // in the test we prefer a deterministic small wait so the ack path is
  // what we're actually measuring.
  std::this_thread::sleep_for(200ms);

  const uint64_t id = manager->migrate(makeRequest(0, 32));
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor([&] {
    return manager->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
  }));
  EXPECT_GE(peer->receivedCount(), 1u);
}

TEST_F(KvmZmqTcpFixture, PerLayerBurstIsFullyAcked) {
  auto cfg = bindTransport();
  ASSERT_TRUE(transport);

  peer = std::make_unique<MockKvManagerPeer>(cfg.cmdEndpoint, cfg.replyEndpoint,
                                             *ctx);
  peer->start();

  manager = std::make_unique<RemoteKVManagerZmqImpl>(std::move(transport), 5s,
                                                     50ms, /*drainPollMs=*/5);

  std::this_thread::sleep_for(200ms);

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
  auto cfg = bindTransport();
  ASSERT_TRUE(transport);

  peer = std::make_unique<MockKvManagerPeer>(cfg.cmdEndpoint, cfg.replyEndpoint,
                                             *ctx, MigrationStatus::FAILED);
  peer->start();

  manager = std::make_unique<RemoteKVManagerZmqImpl>(std::move(transport), 5s,
                                                     50ms, 5);

  std::this_thread::sleep_for(200ms);
  const uint64_t id = manager->migrate(makeRequest(0, 1));
  ASSERT_TRUE(waitFor([&] {
    return manager->getMigrationStatus(id) == MigrationStatus::FAILED;
  }));
}

TEST_F(KvmZmqTcpFixture, TimeoutFiresWhenPeerNeverAcks) {
  auto cfg = bindTransport();
  (void)cfg;
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
