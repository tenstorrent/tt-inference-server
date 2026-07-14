// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/worker/kv_migration_worker.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "messaging/i_kafka_consumer.hpp"
#include "messaging/i_kafka_producer.hpp"
#include "messaging/migration_message.hpp"
#include "runtime/worker/migration_executor.hpp"
#include "runtime/worker/stub_migration_executor.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::worker {
namespace {

using namespace std::chrono_literals;
using tt::messaging::IKafkaConsumer;
using tt::messaging::IKafkaProducer;
using tt::messaging::MigrationRequestMessage;
using tt::messaging::parseMigrationResponse;
using tt::messaging::serialize;
using tt::services::MigrationRequest;
using tt::services::MigrationStatus;

// ---------------------------------------------------------------------------
// In-process fakes
// ---------------------------------------------------------------------------

class FakeProducer : public IKafkaProducer {
 public:
  bool send(std::string_view payload, std::string* errorMessage) override {
    return send(payload, /*partition=*/-1, errorMessage);
  }

  bool send(std::string_view payload, int32_t /*partition*/,
            std::string* errorMessage) override {
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

/**
 * Configurable mock executor.
 *
 *   defaultStatus        - status reported to onDone when no per-id forced
 *                          value is set (initial: SUCCESSFUL).
 *   asyncCallback        - if true, spawns a detached thread that fires
 *                          onDone after a short delay; mimics a real
 *                          asynchronous Mooncake-backed executor.
 *   forceStatusFor(id, s) - per-id override.
 *
 * Captures every (migrationId, request) pair so tests can assert what was
 * dispatched.
 */
class MockMigrationExecutor : public IMigrationExecutor {
 public:
  void execute(uint64_t migrationId, const MigrationRequest& request,
               DoneCallback onDone) override {
    {
      std::lock_guard<std::mutex> lock(mtx);
      submissions.emplace_back(migrationId, request);
    }

    const MigrationStatus status = pickStatus(migrationId);

    if (asyncCallback.load(std::memory_order_relaxed)) {
      std::thread([cb = std::move(onDone), status] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (cb) cb(status);
      }).detach();
    } else {
      if (onDone) onDone(status);
    }
  }

  void setDefaultStatus(MigrationStatus s) {
    std::lock_guard<std::mutex> lock(mtx);
    defaultStatus = s;
  }

  void setAsync(bool enabled) {
    asyncCallback.store(enabled, std::memory_order_relaxed);
  }

  void forceStatusFor(uint64_t id, MigrationStatus s) {
    std::lock_guard<std::mutex> lock(mtx);
    forced[id] = s;
  }

  std::vector<std::pair<uint64_t, MigrationRequest>> getSubmissions() const {
    std::lock_guard<std::mutex> lock(mtx);
    return submissions;
  }

  size_t submissionCount() const {
    std::lock_guard<std::mutex> lock(mtx);
    return submissions.size();
  }

 private:
  MigrationStatus pickStatus(uint64_t id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = forced.find(id);
    return it == forced.end() ? defaultStatus : it->second;
  }

  mutable std::mutex mtx;
  std::vector<std::pair<uint64_t, MigrationRequest>> submissions;
  std::unordered_map<uint64_t, MigrationStatus> forced;
  MigrationStatus defaultStatus = MigrationStatus::SUCCESSFUL;
  std::atomic<bool> asyncCallback{false};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

MigrationRequest makeApiRequest(uint32_t src = 1, uint32_t dst = 2) {
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

std::string serializeReq(uint64_t id, const MigrationRequest& r) {
  return serialize(MigrationRequestMessage{
      .migration_id = id,
      .src_slot = r.src_slot,
      .dst_slot = r.dst_slot,
      .layer_begin = r.layer_begin,
      .layer_end = r.layer_end,
      .src_position_begin = r.src_position_begin,
      .src_position_end = r.src_position_end,
      .dst_position_begin = r.dst_position_begin,
      .dst_position_end = r.dst_position_end,
  });
}

template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

struct Harness {
  FakeConsumer* consumer;
  FakeProducer* producer;
  MockMigrationExecutor* executor;
  std::unique_ptr<KvMigrationWorker> worker;
};

Harness makeHarness() {
  auto consumer = std::make_unique<FakeConsumer>();
  auto producer = std::make_unique<FakeProducer>();
  auto executor = std::make_unique<MockMigrationExecutor>();
  auto* consumerPtr = consumer.get();
  auto* producerPtr = producer.get();
  auto* executorPtr = executor.get();
  auto worker = std::make_unique<KvMigrationWorker>(
      std::move(consumer), std::move(producer), std::move(executor),
      /*pollTimeoutMs=*/5);
  return Harness{consumerPtr, producerPtr, executorPtr, std::move(worker)};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(KvMigrationWorkerTest, ValidRequestDispatchedToExecutor) {
  auto h = makeHarness();
  h.worker->start();

  const auto req = makeApiRequest(7, 9);
  h.consumer->push(serializeReq(/*id=*/100, req));

  ASSERT_TRUE(waitFor([&] { return h.executor->submissionCount() == 1; }));
  auto subs = h.executor->getSubmissions();
  ASSERT_EQ(subs.size(), 1u);
  EXPECT_EQ(subs[0].first, 100u);
  EXPECT_EQ(subs[0].second.src_slot, req.src_slot);
  EXPECT_EQ(subs[0].second.dst_slot, req.dst_slot);
  EXPECT_EQ(subs[0].second.layer_begin, req.layer_begin);
  EXPECT_EQ(subs[0].second.layer_end, req.layer_end);
  EXPECT_EQ(subs[0].second.src_position_begin, req.src_position_begin);
  EXPECT_EQ(subs[0].second.src_position_end, req.src_position_end);
  EXPECT_EQ(subs[0].second.dst_position_begin, req.dst_position_begin);
  EXPECT_EQ(subs[0].second.dst_position_end, req.dst_position_end);
}

TEST(KvMigrationWorkerTest, AckPublishedWithExecutorStatus) {
  auto h = makeHarness();
  h.worker->start();

  h.consumer->push(serializeReq(/*id=*/200, makeApiRequest()));

  ASSERT_TRUE(waitFor([&] { return h.producer->payloadCount() == 1; }));
  auto ack = parseMigrationResponse(h.producer->getPayloads().front());
  ASSERT_TRUE(ack.has_value());
  EXPECT_EQ(ack->migration_id, 200u);
  EXPECT_EQ(ack->status, MigrationStatus::SUCCESSFUL);
}

TEST(KvMigrationWorkerTest, ExecutorFailedStatusPropagatesToAck) {
  auto h = makeHarness();
  h.executor->setDefaultStatus(MigrationStatus::FAILED);
  h.worker->start();

  h.consumer->push(serializeReq(/*id=*/300, makeApiRequest()));

  ASSERT_TRUE(waitFor([&] { return h.producer->payloadCount() == 1; }));
  auto ack = parseMigrationResponse(h.producer->getPayloads().front());
  ASSERT_TRUE(ack.has_value());
  EXPECT_EQ(ack->migration_id, 300u);
  EXPECT_EQ(ack->status, MigrationStatus::FAILED);
}

TEST(KvMigrationWorkerTest, MalformedRequestIsDropped) {
  auto h = makeHarness();
  h.worker->start();

  h.consumer->push("{not valid json");
  h.consumer->push("{}");
  // Real request after the garbage, to prove the loop kept running.
  h.consumer->push(serializeReq(/*id=*/42, makeApiRequest()));

  ASSERT_TRUE(waitFor([&] { return h.executor->submissionCount() == 1; }));
  EXPECT_EQ(h.executor->getSubmissions().front().first, 42u);
  // Two malformed messages produced no executor call and no ack.
  EXPECT_EQ(h.producer->payloadCount(), 1u);
}

TEST(KvMigrationWorkerTest, MultipleSequentialRequestsAllAcked) {
  auto h = makeHarness();
  h.executor->forceStatusFor(2, MigrationStatus::FAILED);
  h.worker->start();

  for (uint64_t id : {1ull, 2ull, 3ull}) {
    h.consumer->push(serializeReq(id, makeApiRequest()));
  }

  ASSERT_TRUE(waitFor([&] { return h.producer->payloadCount() == 3; }));

  std::unordered_map<uint64_t, MigrationStatus> seen;
  for (const auto& payload : h.producer->getPayloads()) {
    auto ack = parseMigrationResponse(payload);
    ASSERT_TRUE(ack.has_value());
    seen[ack->migration_id] = ack->status;
  }
  EXPECT_EQ(seen[1], MigrationStatus::SUCCESSFUL);
  EXPECT_EQ(seen[2], MigrationStatus::FAILED);
  EXPECT_EQ(seen[3], MigrationStatus::SUCCESSFUL);
}

TEST(KvMigrationWorkerTest, AsyncExecutorAckIsStillPublished) {
  auto h = makeHarness();
  h.executor->setAsync(true);
  h.worker->start();

  h.consumer->push(serializeReq(/*id=*/777, makeApiRequest()));

  ASSERT_TRUE(waitFor([&] { return h.producer->payloadCount() == 1; }));
  auto ack = parseMigrationResponse(h.producer->getPayloads().front());
  ASSERT_TRUE(ack.has_value());
  EXPECT_EQ(ack->migration_id, 777u);
  EXPECT_EQ(ack->status, MigrationStatus::SUCCESSFUL);

  // Stop synchronously, before destroying the harness; otherwise an
  // async callback could fire after the worker is gone.
  h.worker->stop();
}

TEST(KvMigrationWorkerTest, AckProducerFailureDoesNotStopWorker) {
  auto h = makeHarness();
  h.producer->setShouldSucceed(false);
  h.worker->start();

  h.consumer->push(serializeReq(/*id=*/1, makeApiRequest()));
  ASSERT_TRUE(waitFor([&] { return h.executor->submissionCount() == 1; }));
  // The send was attempted (recorded) even though it returned false.
  EXPECT_EQ(h.producer->payloadCount(), 1u);

  // Re-enable success and send a second request: the worker is still alive.
  h.producer->setShouldSucceed(true);
  h.consumer->push(serializeReq(/*id=*/2, makeApiRequest()));
  ASSERT_TRUE(waitFor([&] { return h.executor->submissionCount() == 2; }));
  ASSERT_TRUE(waitFor([&] { return h.producer->payloadCount() == 2; }));
}

TEST(KvMigrationWorkerTest, NoExecutorYieldsFailedAck) {
  auto consumer = std::make_unique<FakeConsumer>();
  auto producer = std::make_unique<FakeProducer>();
  auto* consumerPtr = consumer.get();
  auto* producerPtr = producer.get();
  KvMigrationWorker worker(std::move(consumer), std::move(producer),
                           /*executor=*/nullptr, /*pollTimeoutMs=*/5);
  worker.start();

  consumerPtr->push(serializeReq(/*id=*/55, makeApiRequest()));

  ASSERT_TRUE(waitFor([&] { return producerPtr->payloadCount() == 1; }));
  auto ack = parseMigrationResponse(producerPtr->getPayloads().front());
  ASSERT_TRUE(ack.has_value());
  EXPECT_EQ(ack->migration_id, 55u);
  EXPECT_EQ(ack->status, MigrationStatus::FAILED);
}

TEST(KvMigrationWorkerTest, StartStopIsIdempotent) {
  auto h = makeHarness();
  h.worker->start();
  h.worker->start();  // no-op
  h.worker->stop();
  h.worker->stop();  // no-op
  SUCCEED();
}

TEST(KvMigrationWorkerTest, DestructorJoinsCleanly) {
  auto h = makeHarness();
  h.worker->start();
  for (uint64_t id = 1; id <= 5; ++id) {
    h.consumer->push(serializeReq(id, makeApiRequest()));
  }
  ASSERT_TRUE(waitFor([&] { return h.producer->payloadCount() == 5; }));
  h.worker.reset();  // destructor stops the loop and joins
  SUCCEED();
}

// ---------------------------------------------------------------------------
// StubMigrationExecutor direct tests (it's tiny but worth covering)
// ---------------------------------------------------------------------------

TEST(StubMigrationExecutorTest, DefaultReturnsSuccessful) {
  StubMigrationExecutor stub;
  std::optional<MigrationStatus> captured;
  stub.execute(/*id=*/1, makeApiRequest(),
               [&](MigrationStatus s) { captured = s; });
  ASSERT_TRUE(captured.has_value());
  EXPECT_EQ(*captured, MigrationStatus::SUCCESSFUL);
}

TEST(StubMigrationExecutorTest, RespectsConfiguredResult) {
  StubMigrationExecutor stub(MigrationStatus::FAILED);
  std::optional<MigrationStatus> captured;
  stub.execute(/*id=*/2, makeApiRequest(),
               [&](MigrationStatus s) { captured = s; });
  ASSERT_TRUE(captured.has_value());
  EXPECT_EQ(*captured, MigrationStatus::FAILED);
}

TEST(StubMigrationExecutorTest, NullCallbackIsTolerated) {
  StubMigrationExecutor stub;
  // Just must not crash.
  stub.execute(/*id=*/3, makeApiRequest(), /*onDone=*/nullptr);
  SUCCEED();
}

}  // namespace
}  // namespace tt::worker
