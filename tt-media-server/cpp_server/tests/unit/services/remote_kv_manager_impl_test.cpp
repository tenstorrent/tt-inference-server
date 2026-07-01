// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_impl.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "messaging/i_kafka_consumer.hpp"
#include "messaging/i_kafka_producer.hpp"
#include "messaging/migration_message.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::services {
namespace {

using namespace std::chrono_literals;
using tt::messaging::IKafkaConsumer;
using tt::messaging::IKafkaProducer;
using tt::messaging::MigrationResponseMessage;
using tt::messaging::parseMigrationRequest;
using tt::messaging::serialize;

// ---------------------------------------------------------------------------
// In-process fakes
// ---------------------------------------------------------------------------

class FakeProducer : public IKafkaProducer {
 public:
  bool send(std::string_view payload, std::string* errorMessage) override {
    {
      std::lock_guard<std::mutex> lock(mtx);
      payloads.emplace_back(payload);
    }
    if (!shouldSucceed.load(std::memory_order_relaxed)) {
      if (errorMessage) {
        *errorMessage = "fake-producer: forced failure";
      }
      return false;
    }
    return true;
  }

  bool flush(int /*timeoutMs*/, std::string* /*errorMessage*/) override {
    return true;
  }

  std::vector<std::string> getPayloads() const {
    std::lock_guard<std::mutex> lock(mtx);
    return payloads;
  }

  size_t payloadCount() const {
    std::lock_guard<std::mutex> lock(mtx);
    return payloads.size();
  }

  void setShouldSucceed(bool ok) {
    shouldSucceed.store(ok, std::memory_order_relaxed);
  }

 private:
  mutable std::mutex mtx;
  std::vector<std::string> payloads;
  std::atomic<bool> shouldSucceed{true};
};

class FakeConsumer : public IKafkaConsumer {
 public:
  std::optional<std::string> receive(int timeoutMs) override {
    std::unique_lock<std::mutex> lock(mtx);
    if (queue.empty()) {
      // Mimic the real broker: block up to timeoutMs for new data. We cap
      // at 5ms so tests don't sleep needlessly between scripted acks.
      cv.wait_for(lock, std::chrono::milliseconds(std::min(timeoutMs, 5)),
                  [this] { return !queue.empty(); });
      if (queue.empty()) {
        return std::nullopt;
      }
    }
    auto msg = std::move(queue.front());
    queue.pop_front();
    return msg;
  }

  void push(std::string msg) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.push_back(std::move(msg));
    }
    cv.notify_one();
  }

 private:
  std::mutex mtx;
  std::condition_variable cv;
  std::deque<std::string> queue;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

MigrationRequest makeRequest(uint32_t src = 1, uint32_t dst = 2) {
  return MigrationRequest{
      .src_slot = src,
      .dst_slot = dst,
      .layer_id = 3,
      .position_start = 0,
      .position_end = 128,
  };
}

std::string makeAck(uint64_t id, MigrationStatus status) {
  return serialize(MigrationResponseMessage{
      .migration_id = id,
      .status = status,
  });
}

// Spin until `pred()` is true or `timeout` elapses. Returns true on success.
// Used to wait on asynchronous status transitions without sleeping a fixed
// duration that would either flake or slow down the suite.
template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

// Convenience builder: aggressive timings so async transitions resolve in
// the milliseconds range.
std::unique_ptr<RemoteKVManagerImpl> makeManager(
    std::unique_ptr<IKafkaProducer> producer,
    std::unique_ptr<IKafkaConsumer> consumer,
    std::chrono::milliseconds timeout = 500ms,
    std::chrono::milliseconds sweep = 10ms) {
  return std::make_unique<RemoteKVManagerImpl>(
      std::move(producer), std::move(consumer),
      /*migrationWorkerPoolSize=*/1, timeout, sweep,
      /*drainPollMs=*/5);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(RemoteKVManagerImplTest, MigrateReturnsNonZeroIdAndStartsInProgress) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producer), std::move(consumer));

  const uint64_t id = mgr->migrate(makeRequest());

  EXPECT_NE(id, 0u);
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);
}

TEST(RemoteKVManagerImplTest, MigratePublishesRequestPayload) {
  auto producerOwned = std::make_unique<FakeProducer>();
  auto* producer = producerOwned.get();
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producerOwned), std::move(consumer));

  const auto request = makeRequest(/*src=*/7, /*dst=*/9);
  const uint64_t id = mgr->migrate(request);

  ASSERT_EQ(producer->payloadCount(), 1u);
  auto parsed = parseMigrationRequest(producer->getPayloads().front());
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->migration_id, id);
  EXPECT_EQ(parsed->src_slot, request.src_slot);
  EXPECT_EQ(parsed->dst_slot, request.dst_slot);
  EXPECT_EQ(parsed->layer_id, request.layer_id);
  EXPECT_EQ(parsed->position_start, request.position_start);
  EXPECT_EQ(parsed->position_end, request.position_end);
}

TEST(RemoteKVManagerImplTest, MultipleMigratesGetDistinctIds) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producer), std::move(consumer));

  const uint64_t a = mgr->migrate(makeRequest());
  const uint64_t b = mgr->migrate(makeRequest());
  const uint64_t c = mgr->migrate(makeRequest());

  EXPECT_NE(a, b);
  EXPECT_NE(b, c);
  EXPECT_NE(a, c);
}

TEST(RemoteKVManagerImplTest, AckSuccessfulTransitionsStatus) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumerOwned = std::make_unique<FakeConsumer>();
  auto* consumer = consumerOwned.get();
  auto mgr = makeManager(std::move(producer), std::move(consumerOwned));

  const uint64_t id = mgr->migrate(makeRequest());
  consumer->push(makeAck(id, MigrationStatus::SUCCESSFUL));

  ASSERT_TRUE(waitFor(
      [&] { return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL; }));
}

TEST(RemoteKVManagerImplTest, AckFailedTransitionsStatus) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumerOwned = std::make_unique<FakeConsumer>();
  auto* consumer = consumerOwned.get();
  auto mgr = makeManager(std::move(producer), std::move(consumerOwned));

  const uint64_t id = mgr->migrate(makeRequest());
  consumer->push(makeAck(id, MigrationStatus::FAILED));

  ASSERT_TRUE(
      waitFor([&] { return mgr->getMigrationStatus(id) == MigrationStatus::FAILED; }));
}

TEST(RemoteKVManagerImplTest, GetStatusUnknownIdReturnsUnknown) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producer), std::move(consumer));

  EXPECT_EQ(mgr->getMigrationStatus(/*never-issued=*/0xDEADBEEFCAFEBABEull),
            MigrationStatus::UNKNOWN);
}

TEST(RemoteKVManagerImplTest, AckForUnknownIdDoesNotCreateEntry) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumerOwned = std::make_unique<FakeConsumer>();
  auto* consumer = consumerOwned.get();
  auto mgr = makeManager(std::move(producer), std::move(consumerOwned));

  consumer->push(makeAck(/*id=*/12345, MigrationStatus::SUCCESSFUL));
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(mgr->getMigrationStatus(12345), MigrationStatus::UNKNOWN);
}

TEST(RemoteKVManagerImplTest, MalformedAckIsDropped) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumerOwned = std::make_unique<FakeConsumer>();
  auto* consumer = consumerOwned.get();
  auto mgr = makeManager(std::move(producer), std::move(consumerOwned));

  const uint64_t id = mgr->migrate(makeRequest());
  consumer->push("{not valid json");
  consumer->push("{}");

  // Manager should still be processing future acks - send a real one and
  // confirm it lands.
  consumer->push(makeAck(id, MigrationStatus::SUCCESSFUL));
  ASSERT_TRUE(waitFor(
      [&] { return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL; }));
}

TEST(RemoteKVManagerImplTest, SecondAckDoesNotOverwriteTerminalStatus) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumerOwned = std::make_unique<FakeConsumer>();
  auto* consumer = consumerOwned.get();
  auto mgr = makeManager(std::move(producer), std::move(consumerOwned));

  const uint64_t id = mgr->migrate(makeRequest());
  consumer->push(makeAck(id, MigrationStatus::SUCCESSFUL));
  ASSERT_TRUE(waitFor(
      [&] { return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL; }));

  consumer->push(makeAck(id, MigrationStatus::FAILED));
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::SUCCESSFUL);
}

TEST(RemoteKVManagerImplTest, TimeoutSweeperFlipsStaleMigrationsToFailed) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producer), std::move(consumer),
                         /*timeout=*/50ms, /*sweep=*/10ms);

  const uint64_t id = mgr->migrate(makeRequest());
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(
      waitFor([&] { return mgr->getMigrationStatus(id) == MigrationStatus::FAILED; },
              /*timeout=*/1s));
}

TEST(RemoteKVManagerImplTest, AckBeforeTimeoutWins) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumerOwned = std::make_unique<FakeConsumer>();
  auto* consumer = consumerOwned.get();
  auto mgr = makeManager(std::move(producer), std::move(consumerOwned),
                         /*timeout=*/500ms, /*sweep=*/10ms);

  const uint64_t id = mgr->migrate(makeRequest());
  consumer->push(makeAck(id, MigrationStatus::SUCCESSFUL));
  ASSERT_TRUE(waitFor(
      [&] { return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL; }));

  // Wait beyond the timeout: terminal SUCCESSFUL must NOT be flipped.
  std::this_thread::sleep_for(700ms);
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::SUCCESSFUL);
}

TEST(RemoteKVManagerImplTest, SendFailureMarksMigrationFailedImmediately) {
  auto producerOwned = std::make_unique<FakeProducer>();
  auto* producer = producerOwned.get();
  producer->setShouldSucceed(false);
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producerOwned), std::move(consumer));

  const uint64_t id = mgr->migrate(makeRequest());
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::FAILED);
}

TEST(RemoteKVManagerImplTest, ConcurrentMigratesAreThreadSafe) {
  auto producerOwned = std::make_unique<FakeProducer>();
  auto* producer = producerOwned.get();
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producerOwned), std::move(consumer));

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

  EXPECT_EQ(producer->payloadCount(),
            static_cast<size_t>(kThreads * kPerThread));
  for (const auto& ids : idsPerThread) {
    for (uint64_t id : ids) {
      EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);
    }
  }
}

TEST(RemoteKVManagerImplTest, PolymorphicViaIRemoteKVManager) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumerOwned = std::make_unique<FakeConsumer>();
  auto* consumer = consumerOwned.get();
  std::unique_ptr<IRemoteKVManager> mgr =
      makeManager(std::move(producer), std::move(consumerOwned));

  const uint64_t id = mgr->migrate(makeRequest());
  consumer->push(makeAck(id, MigrationStatus::SUCCESSFUL));
  ASSERT_TRUE(waitFor(
      [&] { return mgr->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL; }));
}

TEST(RemoteKVManagerImplTest, DestructorJoinsCleanlyWithPendingMigrations) {
  auto producer = std::make_unique<FakeProducer>();
  auto consumer = std::make_unique<FakeConsumer>();
  auto mgr = makeManager(std::move(producer), std::move(consumer),
                         /*timeout=*/10s, /*sweep=*/5s);

  for (int i = 0; i < 10; ++i) {
    (void)mgr->migrate(makeRequest());
  }
  // No explicit join - destructor must shut the drain thread down even
  // though every migration is still IN_PROGRESS.
  mgr.reset();
  SUCCEED();
}

}  // namespace
}  // namespace tt::services
