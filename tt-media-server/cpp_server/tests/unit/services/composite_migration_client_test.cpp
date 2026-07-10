// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/composite_migration_client.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tt_llm_engine/scheduler/migration_client_interface.hpp>
#include <utility>

namespace tt::services {
namespace {

using MI = tt_llm_engine::scheduler::MigrationClientInterface;
using tt_llm_engine::scheduler::EndpointDisconnectedEvent;
using tt_llm_engine::scheduler::MigrationCompleteEvent;
using tt_llm_engine::scheduler::MigrationFailedEvent;
using tt_llm_engine::scheduler::MigrationReceivedEvent;
using tt_llm_engine::scheduler::MigrationToken;

// ---------------------------------------------------------------------------
// FakeBackend
//
// Minimal MigrationClientInterface implementation that (a) counts every call
// so tests can assert which backend a given method landed on, (b) exposes
// the callbacks the composite registers so tests can verify events fired by
// either backend reach the caller-registered handler, and (c) lets tests
// override return values that the composite must combine (poll sum,
// cmd_queue_write_space min).
// ---------------------------------------------------------------------------
class FakeBackend : public MI {
 public:
  // --- Call counters ---
  int start_burst_calls = 0;
  int enqueue_calls = 0;
  int finish_burst_calls = 0;
  int migrate_calls = 0;
  int poll_calls = 0;
  int connect_to_calls = 0;
  int wait_ready_calls = 0;
  int shutdown_calls = 0;
  int cmd_queue_write_space_calls = 0;

  // --- Configurable behavior ---
  BurstId start_burst_return_value = 0;
  MigrationToken finish_burst_return_value = 0;
  MigrationToken migrate_return_value = 0;
  int poll_return_value = 0;
  uint32_t cmd_queue_write_space_return_value = UINT32_MAX;

  // --- Last-args capture (only fields relevant to routing tests) ---
  MigrationToken last_start_burst_uuid = 0;
  BurstId last_enqueue_burst = 0;
  uint32_t last_migrate_src_slot = 0;
  bool last_shutdown_drain = false;
  std::string last_connect_to_role;
  int last_wait_ready_timeout_ms = 0;

  // --- Registered callbacks (composite mirrors onto both backends) ---
  std::function<void(const MigrationCompleteEvent&)> onComplete;
  std::function<void(const MigrationReceivedEvent&)> onReceived;
  std::function<void(const MigrationFailedEvent&)> onFailed;
  std::function<void(const EndpointDisconnectedEvent&)> onDisconnected;
  std::function<void(int)> onConnectionReceived;

  // --- Burst path ---
  BurstId start_burst(MigrationToken uuid) override {
    ++start_burst_calls;
    last_start_burst_uuid = uuid;
    return start_burst_return_value;
  }

  void enqueue_migration_in_burst(BurstId burst, int /*remote_endpoint_id*/,
                                  uint32_t /*src_slot*/, uint32_t /*dst_slot*/,
                                  uint32_t /*layer_start*/,
                                  uint32_t /*layer_end_exclusive*/,
                                  uint32_t /*pos_start*/,
                                  uint32_t /*pos_end_exclusive*/) override {
    ++enqueue_calls;
    last_enqueue_burst = burst;
  }

  MigrationToken finish_burst(BurstId /*burst*/) override {
    ++finish_burst_calls;
    return finish_burst_return_value;
  }

  // --- Backpressure ---
  uint32_t cmd_queue_write_space() const override {
    // `const` method; keep the counter mutable-updates trivial. Cast away
    // constness on the counter member the same way the sibling adapter
    // would if it tracked state here — this class is a test double and
    // does not otherwise care about thread-safety.
    const_cast<FakeBackend*>(this)->cmd_queue_write_space_calls++;
    return cmd_queue_write_space_return_value;
  }

  // --- Single migrate ---
  MigrationToken migrate(int /*remote_endpoint_id*/, uint32_t src_slot,
                         uint32_t /*dst_slot*/, uint32_t /*layer_start*/,
                         uint32_t /*layer_end_exclusive*/,
                         uint32_t /*pos_start*/,
                         uint32_t /*pos_end_exclusive*/) override {
    ++migrate_calls;
    last_migrate_src_slot = src_slot;
    return migrate_return_value;
  }

  // --- Poll ---
  int poll() override {
    ++poll_calls;
    return poll_return_value;
  }

  // --- Callback registration ---
  void on_migration_complete(
      std::function<void(const MigrationCompleteEvent&)> cb) override {
    onComplete = std::move(cb);
  }

  void on_migration_received(
      std::function<void(const MigrationReceivedEvent&)> cb) override {
    onReceived = std::move(cb);
  }

  void on_migration_failed(
      std::function<void(const MigrationFailedEvent&)> cb) override {
    onFailed = std::move(cb);
  }

  void on_endpoint_disconnected(
      std::function<void(const EndpointDisconnectedEvent&)> cb) override {
    onDisconnected = std::move(cb);
  }

  void on_connection_received(std::function<void(int)> cb) override {
    onConnectionReceived = std::move(cb);
  }

  // --- Lifecycle ---
  void connect_to(int /*remote_endpoint_id*/, const std::string& role,
                  const std::string& /*service_name*/) override {
    ++connect_to_calls;
    last_connect_to_role = role;
  }

  void wait_ready(int timeout_ms) override {
    ++wait_ready_calls;
    last_wait_ready_timeout_ms = timeout_ms;
  }

  void shutdown(bool drain) override {
    ++shutdown_calls;
    last_shutdown_drain = drain;
  }
};

// ---------------------------------------------------------------------------
// Test fixture: wires a fresh composite over two FakeBackends and exposes
// non-owning pointers into them so tests can inspect state after moves.
// ---------------------------------------------------------------------------
struct Wiring {
  std::unique_ptr<CompositeMigrationClient> composite;
  FakeBackend* burst = nullptr;
  FakeBackend* loopback = nullptr;
};

Wiring makeWiring() {
  auto burstOwned = std::make_unique<FakeBackend>();
  auto loopbackOwned = std::make_unique<FakeBackend>();
  Wiring w;
  w.burst = burstOwned.get();
  w.loopback = loopbackOwned.get();
  w.composite = std::make_unique<CompositeMigrationClient>(
      std::move(burstOwned), std::move(loopbackOwned));
  return w;
}

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

TEST(CompositeMigrationClientTest, ConstructorRejectsNullBurstBackend) {
  auto loopback = std::make_unique<FakeBackend>();
  EXPECT_THROW(CompositeMigrationClient(nullptr, std::move(loopback)),
               std::invalid_argument);
}

TEST(CompositeMigrationClientTest, ConstructorRejectsNullLoopbackBackend) {
  auto burst = std::make_unique<FakeBackend>();
  EXPECT_THROW(CompositeMigrationClient(std::move(burst), nullptr),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Burst path -> burst_ only
// ---------------------------------------------------------------------------

TEST(CompositeMigrationClientTest, StartBurstRoutesToBurstOnly) {
  auto w = makeWiring();
  w.burst->start_burst_return_value = 42;

  const auto ret = w.composite->start_burst(/*uuid=*/7);

  EXPECT_EQ(ret, 42u);
  EXPECT_EQ(w.burst->start_burst_calls, 1);
  EXPECT_EQ(w.burst->last_start_burst_uuid, 7u);
  EXPECT_EQ(w.loopback->start_burst_calls, 0);
}

TEST(CompositeMigrationClientTest, EnqueueMigrationInBurstRoutesToBurstOnly) {
  auto w = makeWiring();

  w.composite->enqueue_migration_in_burst(
      /*burst=*/99, /*remote_endpoint_id=*/1, /*src_slot=*/2, /*dst_slot=*/3,
      /*layer_start=*/0, /*layer_end_exclusive=*/61, /*pos_start=*/0,
      /*pos_end_exclusive=*/128);

  EXPECT_EQ(w.burst->enqueue_calls, 1);
  EXPECT_EQ(w.burst->last_enqueue_burst, 99u);
  EXPECT_EQ(w.loopback->enqueue_calls, 0);
}

TEST(CompositeMigrationClientTest, FinishBurstRoutesToBurstOnly) {
  auto w = makeWiring();
  w.burst->finish_burst_return_value = 123;

  const auto tok = w.composite->finish_burst(/*burst=*/5);

  EXPECT_EQ(tok, 123u);
  EXPECT_EQ(w.burst->finish_burst_calls, 1);
  EXPECT_EQ(w.loopback->finish_burst_calls, 0);
}

// ---------------------------------------------------------------------------
// Single migrate -> loopback_ only
// ---------------------------------------------------------------------------

TEST(CompositeMigrationClientTest, MigrateRoutesToLoopbackOnly) {
  auto w = makeWiring();
  w.loopback->migrate_return_value = 555;

  // The whole reason CompositeMigrationClient exists: this call must NOT
  // land on the Kafka burst_ backend (RemoteKVManagerAdapter would throw).
  const auto tok = w.composite->migrate(
      /*remote_endpoint_id=*/0, /*src_slot=*/11, /*dst_slot=*/12,
      /*layer_start=*/0, /*layer_end_exclusive=*/61, /*pos_start=*/0,
      /*pos_end_exclusive=*/64);

  EXPECT_EQ(tok, 555u);
  EXPECT_EQ(w.loopback->migrate_calls, 1);
  EXPECT_EQ(w.loopback->last_migrate_src_slot, 11u);
  EXPECT_EQ(w.burst->migrate_calls, 0);
}

// ---------------------------------------------------------------------------
// Fan-out: poll drains both and returns the sum
// ---------------------------------------------------------------------------

TEST(CompositeMigrationClientTest, PollFansOutAndReturnsSum) {
  auto w = makeWiring();
  w.burst->poll_return_value = 2;
  w.loopback->poll_return_value = 3;

  EXPECT_EQ(w.composite->poll(), 5);
  EXPECT_EQ(w.burst->poll_calls, 1);
  EXPECT_EQ(w.loopback->poll_calls, 1);
}

TEST(CompositeMigrationClientTest, PollReturnsZeroWhenNeitherBackendPolls) {
  auto w = makeWiring();
  // Both default to 0.
  EXPECT_EQ(w.composite->poll(), 0);
  EXPECT_EQ(w.burst->poll_calls, 1);
  EXPECT_EQ(w.loopback->poll_calls, 1);
}

// ---------------------------------------------------------------------------
// Backpressure: cmd_queue_write_space picks the tighter of the two backends
// ---------------------------------------------------------------------------

TEST(CompositeMigrationClientTest, CmdQueueWriteSpaceReturnsMinOfBothBackends) {
  auto w = makeWiring();

  // Loopback (shmem-shaped) tighter than burst (Kafka -> UINT32_MAX).
  w.burst->cmd_queue_write_space_return_value = UINT32_MAX;
  w.loopback->cmd_queue_write_space_return_value = 42;
  EXPECT_EQ(w.composite->cmd_queue_write_space(), 42u);

  // Burst tighter than loopback (defensive: composite must not favor a
  // particular slot; taking the min is what makes the scheduler's
  // kCmdQueueBackpressureMargin throttle keep working).
  w.burst->cmd_queue_write_space_return_value = 7;
  w.loopback->cmd_queue_write_space_return_value = UINT32_MAX;
  EXPECT_EQ(w.composite->cmd_queue_write_space(), 7u);

  // Equal.
  w.burst->cmd_queue_write_space_return_value = 10;
  w.loopback->cmd_queue_write_space_return_value = 10;
  EXPECT_EQ(w.composite->cmd_queue_write_space(), 10u);
}

// ---------------------------------------------------------------------------
// Callback registration: mirror the SAME callback onto both backends so
// events fired by either reach the caller's registered handler.
// ---------------------------------------------------------------------------

TEST(CompositeMigrationClientTest,
     OnMigrationCompleteMirroredAndEventsFromEitherBackendReachCaller) {
  auto w = makeWiring();

  int calls = 0;
  MigrationToken last_token = 0;
  w.composite->on_migration_complete([&](const MigrationCompleteEvent& e) {
    ++calls;
    last_token = e.token;
  });

  // Composite must have registered onto both backends.
  ASSERT_TRUE(static_cast<bool>(w.burst->onComplete));
  ASSERT_TRUE(static_cast<bool>(w.loopback->onComplete));

  // An event from the BURST backend (typical: Kafka burst completes).
  w.burst->onComplete({.token = 100, .ok = true});
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(last_token, 100u);

  // An event from the LOOPBACK backend (typical: shmem migrate() completes
  // — the case that PREFILL_USE_REMOTE_KV_MANAGER=1 would previously have
  // dropped on the floor via a thrown logic_error).
  w.loopback->onComplete({.token = 200, .ok = true});
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(last_token, 200u);
}

TEST(CompositeMigrationClientTest,
     OnMigrationFailedMirroredAndEventsFromEitherBackendReachCaller) {
  auto w = makeWiring();

  int calls = 0;
  w.composite->on_migration_failed(
      [&](const MigrationFailedEvent&) { ++calls; });

  ASSERT_TRUE(static_cast<bool>(w.burst->onFailed));
  ASSERT_TRUE(static_cast<bool>(w.loopback->onFailed));

  w.burst->onFailed({.token = 1, .remote_endpoint_id = 0, .reason = 0});
  w.loopback->onFailed({.token = 2, .remote_endpoint_id = 0, .reason = 0});
  EXPECT_EQ(calls, 2);
}

TEST(CompositeMigrationClientTest, AllOtherCallbacksAreAlsoMirrored) {
  auto w = makeWiring();

  w.composite->on_migration_received([](const MigrationReceivedEvent&) {});
  w.composite->on_endpoint_disconnected(
      [](const EndpointDisconnectedEvent&) {});
  w.composite->on_connection_received([](int) {});

  EXPECT_TRUE(static_cast<bool>(w.burst->onReceived));
  EXPECT_TRUE(static_cast<bool>(w.loopback->onReceived));
  EXPECT_TRUE(static_cast<bool>(w.burst->onDisconnected));
  EXPECT_TRUE(static_cast<bool>(w.loopback->onDisconnected));
  EXPECT_TRUE(static_cast<bool>(w.burst->onConnectionReceived));
  EXPECT_TRUE(static_cast<bool>(w.loopback->onConnectionReceived));
}

// ---------------------------------------------------------------------------
// Lifecycle: connect_to, wait_ready, shutdown fan out to both backends
// ---------------------------------------------------------------------------

TEST(CompositeMigrationClientTest, ConnectToFansOutToBothBackends) {
  auto w = makeWiring();

  w.composite->connect_to(/*remote_endpoint_id=*/1, "PUBLISHER", "ds_pd");

  EXPECT_EQ(w.burst->connect_to_calls, 1);
  EXPECT_EQ(w.loopback->connect_to_calls, 1);
  EXPECT_EQ(w.burst->last_connect_to_role, "PUBLISHER");
  EXPECT_EQ(w.loopback->last_connect_to_role, "PUBLISHER");
}

TEST(CompositeMigrationClientTest, WaitReadyFansOutToBothBackends) {
  auto w = makeWiring();

  w.composite->wait_ready(/*timeout_ms=*/1234);

  EXPECT_EQ(w.burst->wait_ready_calls, 1);
  EXPECT_EQ(w.loopback->wait_ready_calls, 1);
  EXPECT_EQ(w.burst->last_wait_ready_timeout_ms, 1234);
  EXPECT_EQ(w.loopback->last_wait_ready_timeout_ms, 1234);
}

TEST(CompositeMigrationClientTest, ShutdownFansOutToBothBackendsWithDrainFlag) {
  auto w = makeWiring();

  w.composite->shutdown(/*drain=*/true);
  EXPECT_EQ(w.burst->shutdown_calls, 1);
  EXPECT_EQ(w.loopback->shutdown_calls, 1);
  EXPECT_TRUE(w.burst->last_shutdown_drain);
  EXPECT_TRUE(w.loopback->last_shutdown_drain);

  w.composite->shutdown(/*drain=*/false);
  EXPECT_EQ(w.burst->shutdown_calls, 2);
  EXPECT_EQ(w.loopback->shutdown_calls, 2);
  EXPECT_FALSE(w.burst->last_shutdown_drain);
  EXPECT_FALSE(w.loopback->last_shutdown_drain);
}

}  // namespace
}  // namespace tt::services
