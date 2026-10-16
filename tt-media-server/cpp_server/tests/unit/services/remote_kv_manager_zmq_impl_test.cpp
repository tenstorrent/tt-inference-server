// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_zmq_impl.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "messaging/i_kvm_zmq_transport.hpp"
#include "messaging/kvm_command_message.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::services {
namespace {

using namespace std::chrono_literals;
using tt::messaging::IKvmZmqTransport;
using tt::messaging::KvmResponseMessage;
using tt::messaging::parseKvmCommand;
using tt::messaging::serialize;

// ---------------------------------------------------------------------------
// In-process fake transport: records sent payloads, hands back scripted acks.
// Mirrors the FakeProducer/FakeConsumer split from the Kafka impl test but
// collapses them into a single object because IKvmZmqTransport combines
// both directions.
// ---------------------------------------------------------------------------

class FakeTransport : public IKvmZmqTransport {
 public:
  bool send(std::string_view payload, std::string* errorMessage) override {
    {
      std::lock_guard<std::mutex> lock(sentMtx);
      sent.emplace_back(payload);
    }
    if (!shouldSucceed.load(std::memory_order_relaxed)) {
      if (errorMessage) *errorMessage = "fake-transport: forced send failure";
      return false;
    }
    return true;
  }

  std::optional<std::string> receive(int timeoutMs) override {
    std::unique_lock<std::mutex> lock(recvMtx);
    if (recvQueue.empty()) {
      // Cap the wait so tests don't sleep needlessly between scripted acks.
      recvCv.wait_for(lock, std::chrono::milliseconds(std::min(timeoutMs, 5)),
                      [this] { return !recvQueue.empty(); });
      if (recvQueue.empty()) return std::nullopt;
    }
    auto msg = std::move(recvQueue.front());
    recvQueue.pop_front();
    return msg;
  }

  void pushAck(std::string payload) {
    {
      std::lock_guard<std::mutex> lock(recvMtx);
      recvQueue.push_back(std::move(payload));
    }
    recvCv.notify_one();
  }

  std::vector<std::string> getSent() const {
    std::lock_guard<std::mutex> lock(sentMtx);
    return sent;
  }

  size_t sentCount() const {
    std::lock_guard<std::mutex> lock(sentMtx);
    return sent.size();
  }

  void setShouldSucceed(bool ok) {
    shouldSucceed.store(ok, std::memory_order_relaxed);
  }

 private:
  mutable std::mutex sentMtx;
  std::vector<std::string> sent;

  std::mutex recvMtx;
  std::condition_variable recvCv;
  std::deque<std::string> recvQueue;

  std::atomic<bool> shouldSucceed{true};
};

MigrationRequest makeRequest(uint32_t src = 1, uint32_t dst = 2) {
  return MigrationRequest{
      .src_slot = src,
      .dst_slot = dst,
      .layer_begin = 0,
      .layer_end = 32,
      .src_position_begin = 0,
      .src_position_end = 128,
      .dst_position_begin = 0,
      .dst_position_end = 128,
  };
}

std::string makeAck(uint64_t commandId, uint64_t migrationId,
                    MigrationStatus status) {
  return serialize(KvmResponseMessage{
      .command_id = commandId,
      .migration_id = migrationId,
      .status = status,
  });
}

// Spin until `pred()` is true or `timeout` elapses. Used to wait on async
// status transitions without a fixed sleep that either flakes or slows
// down the suite.
template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

std::unique_ptr<RemoteKVManagerZmqImpl> makeManager(
    std::unique_ptr<IKvmZmqTransport> transport,
    std::chrono::milliseconds timeout = 500ms,
    std::chrono::milliseconds sweep = 10ms) {
  return std::make_unique<RemoteKVManagerZmqImpl>(std::move(transport), timeout,
                                                  sweep, /*drainPollMs=*/5);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(RemoteKVManagerZmqImplTest, MigrateReturnsNonZeroIdAndStartsInProgress) {
  auto mgr = makeManager(std::make_unique<FakeTransport>());
  const uint64_t id = mgr->migrate(makeRequest());
  EXPECT_NE(id, 0u);
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);
}

TEST(RemoteKVManagerZmqImplTest, MigratePublishesCommandPayload) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  auto req = makeRequest(/*src=*/7, /*dst=*/9);
  req.migration_id = 1001;
  const uint64_t id = mgr->migrate(req);

  ASSERT_EQ(transport->sentCount(), 1u);
  auto parsed = parseKvmCommand(transport->getSent().front());
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->command_id, id);
  EXPECT_EQ(parsed->migration_id, 1001u);
  EXPECT_EQ(parsed->src_slot, req.src_slot);
  EXPECT_EQ(parsed->dst_slot, req.dst_slot);
  EXPECT_EQ(parsed->layer_begin, req.layer_begin);
  EXPECT_EQ(parsed->layer_end, req.layer_end);
}

TEST(RemoteKVManagerZmqImplTest, MissingMigrationIdFallsBackToCommandId) {
  // The adapter always sets migration_id, but the impl must survive
  // callers that don't so downstream logs still have a stable key. We
  // fall back to command_id in that case.
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  auto req = makeRequest();
  req.migration_id = std::nullopt;
  const uint64_t id = mgr->migrate(req);

  auto parsed = parseKvmCommand(transport->getSent().front());
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->command_id, id);
  EXPECT_EQ(parsed->migration_id, id);
}

TEST(RemoteKVManagerZmqImplTest, MultipleMigratesGetDistinctIds) {
  auto mgr = makeManager(std::make_unique<FakeTransport>());
  const uint64_t a = mgr->migrate(makeRequest());
  const uint64_t b = mgr->migrate(makeRequest());
  const uint64_t c = mgr->migrate(makeRequest());
  EXPECT_NE(a, b);
  EXPECT_NE(b, c);
  EXPECT_NE(a, c);
}

TEST(RemoteKVManagerZmqImplTest, AckSuccessfulTransitionsStatus) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  const uint64_t id = mgr->migrate(makeRequest());
  transport->pushAck(makeAck(id, /*migId=*/id, MigrationStatus::SUCCESSFUL));

  ASSERT_TRUE(waitFor([&] {
    return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
  }));
}

TEST(RemoteKVManagerZmqImplTest, AckFailedTransitionsStatus) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  const uint64_t id = mgr->migrate(makeRequest());
  transport->pushAck(makeAck(id, id, MigrationStatus::FAILED));

  ASSERT_TRUE(waitFor(
      [&] { return mgr->getMigrationStatus(id) == MigrationStatus::FAILED; }));
}

TEST(RemoteKVManagerZmqImplTest, GetStatusUnknownIdReturnsUnknown) {
  auto mgr = makeManager(std::make_unique<FakeTransport>());
  EXPECT_EQ(mgr->getMigrationStatus(0xDEADBEEFCAFEBABEull),
            MigrationStatus::UNKNOWN);
}

TEST(RemoteKVManagerZmqImplTest, AckForUnknownIdDoesNotCreateEntry) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  transport->pushAck(makeAck(12345, 12345, MigrationStatus::SUCCESSFUL));
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(mgr->getMigrationStatus(12345), MigrationStatus::UNKNOWN);
}

TEST(RemoteKVManagerZmqImplTest, MalformedAckIsDropped) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  const uint64_t id = mgr->migrate(makeRequest());
  transport->pushAck("{not valid json");
  transport->pushAck("{}");
  transport->pushAck(makeAck(id, id, MigrationStatus::SUCCESSFUL));

  ASSERT_TRUE(waitFor([&] {
    return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
  }));
}

TEST(RemoteKVManagerZmqImplTest, SecondAckDoesNotOverwriteTerminalStatus) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  const uint64_t id = mgr->migrate(makeRequest());
  transport->pushAck(makeAck(id, id, MigrationStatus::SUCCESSFUL));
  ASSERT_TRUE(waitFor([&] {
    return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
  }));

  transport->pushAck(makeAck(id, id, MigrationStatus::FAILED));
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::SUCCESSFUL);
}

TEST(RemoteKVManagerZmqImplTest, TimeoutSweeperFlipsStaleMigrationsToFailed) {
  auto mgr = makeManager(std::make_unique<FakeTransport>(),
                         /*timeout=*/50ms, /*sweep=*/10ms);

  const uint64_t id = mgr->migrate(makeRequest());
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(waitFor(
      [&] { return mgr->getMigrationStatus(id) == MigrationStatus::FAILED; },
      /*timeout=*/1s));
}

TEST(RemoteKVManagerZmqImplTest, AckBeforeTimeoutWins) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned),
                         /*timeout=*/500ms, /*sweep=*/10ms);

  const uint64_t id = mgr->migrate(makeRequest());
  transport->pushAck(makeAck(id, id, MigrationStatus::SUCCESSFUL));
  ASSERT_TRUE(waitFor([&] {
    return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
  }));

  std::this_thread::sleep_for(700ms);
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::SUCCESSFUL);
}

TEST(RemoteKVManagerZmqImplTest, SendFailureMarksMigrationFailedImmediately) {
  auto owned = std::make_unique<FakeTransport>();
  owned->setShouldSucceed(false);
  auto mgr = makeManager(std::move(owned));

  const uint64_t id = mgr->migrate(makeRequest());
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::FAILED);
}

TEST(RemoteKVManagerZmqImplTest, ConcurrentMigratesAreThreadSafe) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  auto mgr = makeManager(std::move(owned));

  constexpr int kThreads = 8;
  constexpr int kPerThread = 50;
  std::vector<std::thread> threads;
  std::vector<std::vector<uint64_t>> idsPerThread(kThreads);
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      idsPerThread[t].reserve(kPerThread);
      for (int i = 0; i < kPerThread; ++i) {
        idsPerThread[t].push_back(mgr->migrate(makeRequest()));
      }
    });
  }
  for (auto& th : threads) th.join();

  EXPECT_EQ(transport->sentCount(), static_cast<size_t>(kThreads * kPerThread));
  for (const auto& ids : idsPerThread) {
    for (uint64_t id : ids) {
      EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);
    }
  }
}

TEST(RemoteKVManagerZmqImplTest, PolymorphicViaIRemoteKVManager) {
  auto owned = std::make_unique<FakeTransport>();
  auto* transport = owned.get();
  std::unique_ptr<IRemoteKVManager> mgr = makeManager(std::move(owned));

  const uint64_t id = mgr->migrate(makeRequest());
  transport->pushAck(makeAck(id, id, MigrationStatus::SUCCESSFUL));
  ASSERT_TRUE(waitFor([&] {
    return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
  }));
}

TEST(RemoteKVManagerZmqImplTest, DestructorJoinsCleanlyWithPendingMigrations) {
  auto mgr = makeManager(std::make_unique<FakeTransport>(),
                         /*timeout=*/10s, /*sweep=*/5s);

  for (int i = 0; i < 10; ++i) {
    (void)mgr->migrate(makeRequest());
  }
  mgr.reset();
  SUCCEED();
}

}  // namespace
}  // namespace tt::services
