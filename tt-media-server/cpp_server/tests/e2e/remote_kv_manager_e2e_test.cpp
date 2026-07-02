// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// End-to-end test for RemoteKVManagerImpl against a real Kafka broker.
//
// Uses the production RemoteKVManagerImpl (real KafkaProducer / KafkaConsumer
// + real JSON messaging) on one side, and a controllable MockMigrationWorker
// on the other side. The mock consumes request messages via a real
// KafkaConsumer and publishes acks via a real KafkaProducer; its reply
// behavior (status, delay, drop) is scriptable at runtime so tests can drive
// the manager through success / failure / timeout scenarios.
//
// Requires the dev Kafka broker to be up (scripts/dev-kafka.sh up). The
// broker address defaults to "kafka:9092" (docker network DNS) and can be
// overridden via the KAFKA_BROKERS env var. Skipped when the broker is
// unreachable so a plain `ctest` run without a broker stays green.

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "messaging/migration_message.hpp"
#include "services/remote_kv_manager.hpp"
#include "services/remote_kv_manager_impl.hpp"
#include "utils/logger.hpp"

namespace tt::services {
namespace {

using namespace std::chrono_literals;

// Warmup covers three things happening back-to-back:
//   1. Broker auto-creates the (fresh, unique) request/ack topics on first
//      subscribe -- can take a few seconds under Kafka 4.0.
//   2. Both consumer groups (manager ack-consumer, worker request-consumer)
//      finish partition assignment.
//   3. Their fetchers reach the latest offset so a subsequent produce isn't
//      silently skipped by auto.offset.reset=latest.
// 8s is comfortably above what we observe locally; tune down only if
// auto-creation latency improves.
constexpr auto KAFKA_GROUP_JOIN_WARMUP = 8s;

// Cadence a caller (e.g. Dynamo migration engine) might use when polling
// getMigrationStatus() while waiting for a terminal state.
constexpr auto POLL_INTERVAL = 50ms;

// Upper bound on how long a single migration is allowed to sit in
// IN_PROGRESS before the test considers it stuck. Covers the mock's reply
// delay plus a generous ack-propagation margin.
constexpr auto POLL_DEADLINE = 5s;

std::string envOr(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : std::string(fallback);
}

std::string uniqueSuffix() {
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  return std::to_string(::getpid()) + "-" + std::to_string(us);
}

template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(20ms);
  }
  return pred();
}

// ---------------------------------------------------------------------------
// MockMigrationWorker
// ---------------------------------------------------------------------------
//
// Real-Kafka-backed drop-in for tt::worker::KvMigrationWorker whose reply
// behavior is scriptable at runtime. Each received MigrationRequestMessage is
// answered with a MigrationResponseMessage carrying the currently configured
// status, after the configured delay -- unless `dropRequest` is set, in which
// case the request is silently swallowed to simulate a worker crash / lost
// ack (i.e. force the manager's timeout sweeper to fire).
class MockMigrationWorker {
 public:
  struct Behavior {
    MigrationStatus replyStatus{MigrationStatus::SUCCESSFUL};
    std::chrono::milliseconds replyDelay{0};
    bool dropRequest{false};
  };

  MockMigrationWorker(const std::string& brokers,
                      const std::string& requestTopic,
                      const std::string& ackTopic, const std::string& groupId) {
    requestConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{
            .brokers = brokers, .topic = requestTopic, .group_id = groupId});
    ackProducer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = ackTopic});
  }

  ~MockMigrationWorker() { stop(); }

  MockMigrationWorker(const MockMigrationWorker&) = delete;
  MockMigrationWorker& operator=(const MockMigrationWorker&) = delete;

  void setBehavior(Behavior b) {
    std::lock_guard<std::mutex> lock(mtx);
    behavior = b;
  }

  Behavior getBehavior() const {
    std::lock_guard<std::mutex> lock(mtx);
    return behavior;
  }

  void start() {
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true)) return;
    thread = std::thread([this] { run(); });
  }

  void stop() {
    bool expected = true;
    if (!running.compare_exchange_strong(expected, false)) return;
    if (thread.joinable()) thread.join();
  }

  std::size_t requestsReceived() const {
    return received.load(std::memory_order_relaxed);
  }

 private:
  void run() {
    while (running.load(std::memory_order_relaxed)) {
      auto raw = requestConsumer->receive(50);
      if (!raw.has_value()) continue;

      auto parsed = tt::messaging::parseMigrationRequest(*raw);
      if (!parsed.has_value()) continue;

      received.fetch_add(1, std::memory_order_relaxed);

      const Behavior b = getBehavior();
      if (b.dropRequest) continue;

      if (b.replyDelay.count() > 0) {
        std::this_thread::sleep_for(b.replyDelay);
      }

      const tt::messaging::MigrationResponseMessage response{
          .migration_id = parsed->migration_id, .status = b.replyStatus};
      const std::string payload = tt::messaging::serialize(response);
      std::string err;
      ackProducer->send(payload, &err);
    }
  }

  std::unique_ptr<tt::messaging::KafkaConsumer> requestConsumer;
  std::unique_ptr<tt::messaging::KafkaProducer> ackProducer;
  std::atomic<bool> running{false};
  std::thread thread;
  mutable std::mutex mtx;
  Behavior behavior;
  std::atomic<std::size_t> received{0};
};

// ---------------------------------------------------------------------------
// MockDownloadWorker
// ---------------------------------------------------------------------------
//
// Same shape as MockMigrationWorker but wired to the download topics and
// answers with DownloadResponseMessage. `downloadedBlockHashes` is echoed
// verbatim into the ack so scenarios can assert the hashes propagate back
// through RemoteKVManagerImpl to getDownloadResult().
class MockDownloadWorker {
 public:
  struct Behavior {
    MigrationStatus replyStatus{MigrationStatus::SUCCESSFUL};
    std::chrono::milliseconds replyDelay{0};
    bool dropRequest{false};
    std::vector<uint64_t> downloadedBlockHashes{};
  };

  MockDownloadWorker(const std::string& brokers,
                     const std::string& requestTopic,
                     const std::string& ackTopic, const std::string& groupId) {
    requestConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{
            .brokers = brokers, .topic = requestTopic, .group_id = groupId});
    ackProducer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = ackTopic});
  }

  ~MockDownloadWorker() { stop(); }

  MockDownloadWorker(const MockDownloadWorker&) = delete;
  MockDownloadWorker& operator=(const MockDownloadWorker&) = delete;

  void setBehavior(Behavior b) {
    std::lock_guard<std::mutex> lock(mtx);
    behavior = std::move(b);
  }

  Behavior getBehavior() const {
    std::lock_guard<std::mutex> lock(mtx);
    return behavior;
  }

  void start() {
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true)) return;
    thread = std::thread([this] { run(); });
  }

  void stop() {
    bool expected = true;
    if (!running.compare_exchange_strong(expected, false)) return;
    if (thread.joinable()) thread.join();
  }

  std::size_t requestsReceived() const {
    return received.load(std::memory_order_relaxed);
  }

 private:
  void run() {
    while (running.load(std::memory_order_relaxed)) {
      auto raw = requestConsumer->receive(50);
      if (!raw.has_value()) continue;

      auto parsed = tt::messaging::parseDownloadRequest(*raw);
      if (!parsed.has_value()) continue;

      received.fetch_add(1, std::memory_order_relaxed);

      const Behavior b = getBehavior();
      if (b.dropRequest) continue;

      if (b.replyDelay.count() > 0) {
        std::this_thread::sleep_for(b.replyDelay);
      }

      const tt::messaging::DownloadResponseMessage response{
          .id = parsed->id,
          .status = b.replyStatus,
          .downloaded_block_hashes =
              b.replyStatus == MigrationStatus::SUCCESSFUL
                  ? b.downloadedBlockHashes
                  : std::vector<uint64_t>{},
      };
      const std::string payload = tt::messaging::serialize(response);
      std::string err;
      ackProducer->send(payload, &err);
    }
  }

  std::unique_ptr<tt::messaging::KafkaConsumer> requestConsumer;
  std::unique_ptr<tt::messaging::KafkaProducer> ackProducer;
  std::atomic<bool> running{false};
  std::thread thread;
  mutable std::mutex mtx;
  Behavior behavior;
  std::atomic<std::size_t> received{0};
};

// ---------------------------------------------------------------------------
// MockOffloadWorker
// ---------------------------------------------------------------------------
//
// Same shape as MockMigrationWorker but wired to the offload topics. Each
// received OffloadRequestMessage is answered with an OffloadResponseMessage
// echoing the migration id and the currently configured status/delay.
// `dropRequest` swallows the request to simulate a lost ack.
class MockOffloadWorker {
 public:
  struct Behavior {
    MigrationStatus replyStatus{MigrationStatus::SUCCESSFUL};
    std::chrono::milliseconds replyDelay{0};
    bool dropRequest{false};
  };

  MockOffloadWorker(const std::string& brokers,
                    const std::string& requestTopic,
                    const std::string& ackTopic, const std::string& groupId) {
    requestConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{
            .brokers = brokers, .topic = requestTopic, .group_id = groupId});
    ackProducer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = ackTopic});
  }

  ~MockOffloadWorker() { stop(); }

  MockOffloadWorker(const MockOffloadWorker&) = delete;
  MockOffloadWorker& operator=(const MockOffloadWorker&) = delete;

  void setBehavior(Behavior b) {
    std::lock_guard<std::mutex> lock(mtx);
    behavior = b;
  }

  Behavior getBehavior() const {
    std::lock_guard<std::mutex> lock(mtx);
    return behavior;
  }

  void start() {
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true)) return;
    thread = std::thread([this] { run(); });
  }

  void stop() {
    bool expected = true;
    if (!running.compare_exchange_strong(expected, false)) return;
    if (thread.joinable()) thread.join();
  }

  std::size_t requestsReceived() const {
    return received.load(std::memory_order_relaxed);
  }

  std::vector<tt::messaging::OffloadRequestMessage> takeReceived() {
    std::lock_guard<std::mutex> lock(mtx);
    auto out = std::move(receivedMessages);
    receivedMessages.clear();
    return out;
  }

 private:
  void run() {
    while (running.load(std::memory_order_relaxed)) {
      auto raw = requestConsumer->receive(50);
      if (!raw.has_value()) continue;

      auto parsed = tt::messaging::parseOffloadRequest(*raw);
      if (!parsed.has_value()) continue;

      received.fetch_add(1, std::memory_order_relaxed);
      const uint64_t id = parsed->id;
      {
        std::lock_guard<std::mutex> lock(mtx);
        receivedMessages.push_back(std::move(*parsed));
      }

      const Behavior b = getBehavior();
      if (b.dropRequest) continue;

      if (b.replyDelay.count() > 0) {
        std::this_thread::sleep_for(b.replyDelay);
      }

      const tt::messaging::OffloadResponseMessage response{
          .id = id, .status = b.replyStatus};
      const std::string payload = tt::messaging::serialize(response);
      std::string err;
      ackProducer->send(payload, &err);
    }
  }

  std::unique_ptr<tt::messaging::KafkaConsumer> requestConsumer;
  std::unique_ptr<tt::messaging::KafkaProducer> ackProducer;
  std::atomic<bool> running{false};
  std::thread thread;
  mutable std::mutex mtx;
  Behavior behavior;
  std::vector<tt::messaging::OffloadRequestMessage> receivedMessages;
  std::atomic<std::size_t> received{0};
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class RemoteKVManagerE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Force logger init on the main thread. RemoteKVManagerImpl spawns its
    // drain thread before its own TT_LOG_INFO, so without this the drain
    // thread and main thread race to spdlog::register_logger and one throws.
    tt::utils::ZeroOverheadLogger::initialize();
  }

  void SetUp() override {
    brokers = envOr("KAFKA_BROKERS", "kafka:9092");

    const auto suffix = uniqueSuffix();
    requestTopic = "e2e-kv-req-" + suffix;
    ackTopic = "e2e-kv-ack-" + suffix;
    downloadRequestTopic = "e2e-kv-dl-req-" + suffix;
    downloadAckTopic = "e2e-kv-dl-ack-" + suffix;
    offloadRequestTopic = "e2e-kv-of-req-" + suffix;
    offloadAckTopic = "e2e-kv-of-ack-" + suffix;
    clientGroup = "e2e-client-" + suffix;
    workerGroup = "e2e-worker-" + suffix;
    downloadWorkerGroup = "e2e-dl-worker-" + suffix;
    offloadWorkerGroup = "e2e-of-worker-" + suffix;
  }

  std::unique_ptr<RemoteKVManagerImpl> makeManager(
      std::chrono::milliseconds timeout = 30s) {
    auto producer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = requestTopic});
    auto consumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{
            .brokers = brokers, .topic = ackTopic, .group_id = clientGroup});
    auto downloadProducer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = downloadRequestTopic});
    auto downloadConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{.brokers = brokers,
                                           .topic = downloadAckTopic,
                                           .group_id = clientGroup + "-dl"});
    auto offloadProducer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = offloadRequestTopic});
    auto offloadConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{.brokers = brokers,
                                           .topic = offloadAckTopic,
                                           .group_id = clientGroup + "-of"});
    return std::make_unique<RemoteKVManagerImpl>(
        std::move(producer), std::move(consumer),
        /*migrationWorkerPoolSize=*/1, timeout,
        /*sweepInterval=*/200ms, /*drainPollMs=*/50,
        std::move(downloadProducer), std::move(downloadConsumer),
        std::move(offloadProducer), std::move(offloadConsumer));
  }

  std::unique_ptr<MockMigrationWorker> makeWorker() {
    return std::make_unique<MockMigrationWorker>(brokers, requestTopic,
                                                 ackTopic, workerGroup);
  }

  std::unique_ptr<MockDownloadWorker> makeDownloadWorker() {
    return std::make_unique<MockDownloadWorker>(brokers, downloadRequestTopic,
                                                downloadAckTopic,
                                                downloadWorkerGroup);
  }

  std::unique_ptr<MockOffloadWorker> makeOffloadWorker() {
    return std::make_unique<MockOffloadWorker>(
        brokers, offloadRequestTopic, offloadAckTopic, offloadWorkerGroup);
  }

  static MigrationRequest sampleRequest() {
    return MigrationRequest{.src_slot = 1,
                            .dst_slot = 2,
                            .layer_id = 3,
                            .position_start = 0,
                            .position_end = 128};
  }

  static DownloadKVRequest sampleDownloadRequest() {
    return DownloadKVRequest{
        .dstSlot = 4,
        .blocks =
            {
                KVCacheBlockRef{
                    .blockHash = 0xAAAAAAAA, .positionId = 0, .tokenCount = 32},
                KVCacheBlockRef{.blockHash = 0xBBBBBBBB,
                                .positionId = 32,
                                .tokenCount = 32},
            },
    };
  }

  static OffloadKVRequest sampleOffloadRequest() {
    return OffloadKVRequest{
        .srcSlot = 7,
        .blocks =
            {
                KVCacheBlockRef{
                    .blockHash = 0xC0FFEE, .positionId = 0, .tokenCount = 32},
            },
    };
  }

  std::string brokers;
  std::string requestTopic;
  std::string ackTopic;
  std::string downloadRequestTopic;
  std::string downloadAckTopic;
  std::string offloadRequestTopic;
  std::string offloadAckTopic;
  std::string clientGroup;
  std::string workerGroup;
  std::string downloadWorkerGroup;
  std::string offloadWorkerGroup;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(RemoteKVManagerE2ETest, MigrateSucceedsAfterOneSecondDelay) {
  auto manager = makeManager();
  auto worker = makeWorker();
  worker->setBehavior({
      .replyStatus = MigrationStatus::SUCCESSFUL,
      .replyDelay = 1s,
  });
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const auto issuedAt = std::chrono::steady_clock::now();
  const uint64_t id = manager->migrate(sampleRequest());
  ASSERT_NE(id, 0u);

  // Simulate a caller polling every POLL_INTERVAL while the mock still
  // sleeps its 1s reply delay: all three samples should be IN_PROGRESS.
  EXPECT_EQ(manager->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);
  std::this_thread::sleep_for(POLL_INTERVAL);
  EXPECT_EQ(manager->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);
  std::this_thread::sleep_for(POLL_INTERVAL);
  EXPECT_EQ(manager->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
      },
      POLL_DEADLINE))
      << "migration did not reach SUCCESSFUL within " << POLL_DEADLINE.count()
      << "ms";

  const auto elapsed = std::chrono::steady_clock::now() - issuedAt;
  EXPECT_GE(elapsed, 900ms)
      << "manager transitioned to SUCCESSFUL before the mock's 1s delay -- "
         "delay was not honored";

  EXPECT_EQ(worker->requestsReceived(), 1u);

  worker->stop();
}

TEST_F(RemoteKVManagerE2ETest, DroppedRequestFailsViaTimeoutSweeper) {
  // Configure the manager with a short timeout so the sweeper flips the
  // migration to FAILED without dragging the test past POLL_DEADLINE.
  constexpr auto managerTimeout = 1s;
  auto manager = makeManager(managerTimeout);

  auto worker = makeWorker();
  worker->setBehavior({.dropRequest = true});
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->migrate(sampleRequest());
  ASSERT_NE(id, 0u);
  EXPECT_EQ(manager->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getMigrationStatus(id) == MigrationStatus::FAILED;
      },
      POLL_DEADLINE))
      << "sweeper never marked the dropped migration FAILED within "
      << POLL_DEADLINE.count() << "ms";

  // The mock did consume the request off the request topic, it just chose
  // not to publish an ack -- so from the manager's POV this simulates a
  // worker that crashed mid-flight.
  EXPECT_EQ(worker->requestsReceived(), 1u);

  worker->stop();
}

TEST_F(RemoteKVManagerE2ETest, WorkerReplyFailedPropagates) {
  auto manager = makeManager();
  auto worker = makeWorker();
  worker->setBehavior({
      .replyStatus = MigrationStatus::FAILED,
      .replyDelay = 0ms,
  });
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->migrate(sampleRequest());
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getMigrationStatus(id) == MigrationStatus::FAILED;
      },
      POLL_DEADLINE))
      << "FAILED ack did not propagate within " << POLL_DEADLINE.count()
      << "ms";

  EXPECT_EQ(worker->requestsReceived(), 1u);

  worker->stop();
}

// ---------------------------------------------------------------------------
// Download scenarios
// ---------------------------------------------------------------------------

TEST_F(RemoteKVManagerE2ETest, DownloadSucceedsAndHashesPropagate) {
  auto manager = makeManager();
  auto worker = makeDownloadWorker();
  const std::vector<uint64_t> expectedHashes{0xAAAAAAAA, 0xBBBBBBBB};
  worker->setBehavior({
      .replyStatus = MigrationStatus::SUCCESSFUL,
      .replyDelay = 500ms,
      .downloadedBlockHashes = expectedHashes,
  });
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->downloadFromStore(sampleDownloadRequest());
  ASSERT_NE(id, 0u);
  EXPECT_EQ(manager->getDownloadResult(id).status,
            MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getDownloadResult(id).status ==
               MigrationStatus::SUCCESSFUL;
      },
      POLL_DEADLINE))
      << "download did not reach SUCCESSFUL within " << POLL_DEADLINE.count()
      << "ms";

  const auto result = manager->getDownloadResult(id);
  EXPECT_EQ(result.status, MigrationStatus::SUCCESSFUL);
  EXPECT_EQ(result.downloadedBlockHashes, expectedHashes);
  EXPECT_EQ(worker->requestsReceived(), 1u);

  worker->stop();
}

TEST_F(RemoteKVManagerE2ETest, DownloadDroppedFailsViaTimeoutSweeper) {
  constexpr auto managerTimeout = 1s;
  auto manager = makeManager(managerTimeout);
  auto worker = makeDownloadWorker();
  worker->setBehavior({.dropRequest = true});
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->downloadFromStore(sampleDownloadRequest());
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getDownloadResult(id).status ==
               MigrationStatus::FAILED;
      },
      POLL_DEADLINE))
      << "sweeper never marked the dropped download FAILED within "
      << POLL_DEADLINE.count() << "ms";

  EXPECT_TRUE(manager->getDownloadResult(id).downloadedBlockHashes.empty());
  EXPECT_EQ(worker->requestsReceived(), 1u);

  worker->stop();
}

TEST_F(RemoteKVManagerE2ETest, DownloadReplyFailedPropagates) {
  auto manager = makeManager();
  auto worker = makeDownloadWorker();
  worker->setBehavior({
      .replyStatus = MigrationStatus::FAILED,
      .replyDelay = 0ms,
  });
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->downloadFromStore(sampleDownloadRequest());
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getDownloadResult(id).status ==
               MigrationStatus::FAILED;
      },
      POLL_DEADLINE))
      << "FAILED download-ack did not propagate within "
      << POLL_DEADLINE.count() << "ms";

  EXPECT_TRUE(manager->getDownloadResult(id).downloadedBlockHashes.empty());
  EXPECT_EQ(worker->requestsReceived(), 1u);

  worker->stop();
}

// ---------------------------------------------------------------------------
// Offload scenarios
// ---------------------------------------------------------------------------

TEST_F(RemoteKVManagerE2ETest, OffloadSucceedsAfterOneSecondDelay) {
  auto manager = makeManager();
  auto worker = makeOffloadWorker();
  worker->setBehavior({
      .replyStatus = MigrationStatus::SUCCESSFUL,
      .replyDelay = 1s,
  });
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const auto req = sampleOffloadRequest();
  const auto issuedAt = std::chrono::steady_clock::now();
  const uint64_t id = manager->offloadToStore(req);
  ASSERT_NE(id, 0u);

  // While the mock still sleeps its 1s reply delay the manager should
  // report IN_PROGRESS (offload is no longer fire-and-forget).
  EXPECT_EQ(manager->getOffloadStatus(id), MigrationStatus::IN_PROGRESS);
  std::this_thread::sleep_for(POLL_INTERVAL);
  EXPECT_EQ(manager->getOffloadStatus(id), MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getOffloadStatus(id) == MigrationStatus::SUCCESSFUL;
      },
      POLL_DEADLINE))
      << "offload did not reach SUCCESSFUL within " << POLL_DEADLINE.count()
      << "ms";

  const auto elapsed = std::chrono::steady_clock::now() - issuedAt;
  EXPECT_GE(elapsed, 900ms)
      << "manager transitioned to SUCCESSFUL before the mock's 1s delay -- "
         "delay was not honored";

  EXPECT_EQ(worker->requestsReceived(), 1u);

  const auto received = worker->takeReceived();
  ASSERT_EQ(received.size(), 1u);
  EXPECT_EQ(received[0].id, id);
  EXPECT_EQ(received[0].src_slot, req.srcSlot);
  ASSERT_EQ(received[0].blocks.size(), req.blocks.size());
  EXPECT_EQ(received[0].blocks[0].blockHash, req.blocks[0].blockHash);

  worker->stop();
}

TEST_F(RemoteKVManagerE2ETest, OffloadReplyFailedPropagates) {
  auto manager = makeManager();
  auto worker = makeOffloadWorker();
  worker->setBehavior({
      .replyStatus = MigrationStatus::FAILED,
      .replyDelay = 0ms,
  });
  worker->start();

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->offloadToStore(sampleOffloadRequest());
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor(
      [&] { return manager->getOffloadStatus(id) == MigrationStatus::FAILED; },
      POLL_DEADLINE))
      << "FAILED offload-ack did not propagate within "
      << POLL_DEADLINE.count() << "ms";

  EXPECT_EQ(worker->requestsReceived(), 1u);

  worker->stop();
}

}  // namespace
}  // namespace tt::services
