// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// End-to-end test for RemoteKVManagerImpl against a real Kafka broker.
//
// Uses the production RemoteKVManagerImpl (real KafkaProducer / KafkaConsumer
// + real JSON messaging) on one side, and a controllable MockKafkaWorker on
// the other side. The mock consumes request messages via a real
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

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "messaging/utils/kafka_utils.hpp"
#include "mock_kafka_worker.hpp"
#include "services/remote_kv_manager.hpp"
#include "services/remote_kv_manager_impl.hpp"
#include "utils/logger.hpp"

namespace tt::services {
namespace {

using namespace std::chrono_literals;
using namespace tt::services::testing;

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
    clientGroup = "e2e-client-" + suffix;
    workerGroup = "e2e-worker-" + suffix;
  }

  std::unique_ptr<RemoteKVManagerImpl> makeManager(
      std::chrono::milliseconds timeout = 30s,
      RemoteKVManagerImpl::LayerToPartition layerToPartition = nullptr) {
    auto producer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = requestTopic});
    auto consumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{
            .brokers = brokers, .topic = ackTopic, .group_id = clientGroup});
    return std::make_unique<RemoteKVManagerImpl>(
        std::move(producer), std::move(consumer), timeout,
        /*sweepInterval=*/200ms, /*drainPollMs=*/50,
        std::move(layerToPartition));
  }

  // Ensure migration req + ack topics exist with the requested partition
  // count. Kafka's default auto-creation gives topics one partition; call
  // this before instantiating a manager or workers that expect to pin to
  // specific partitions.
  bool ensureMigrationTopicsWithPartitions(int32_t numPartitions) {
    return tt::messaging::kafka_utils::createTopicWithPartitions(
               brokers, requestTopic, numPartitions) &&
           tt::messaging::kafka_utils::createTopicWithPartitions(
               brokers, ackTopic, numPartitions);
  }

  std::unique_ptr<MockKafkaWorker> makeWorker() {
    return std::make_unique<MockKafkaWorker>(brokers, requestTopic, ackTopic,
                                             workerGroup, migrationParser(),
                                             migrationResponder());
  }

  // Spin up N migration workers, each in its own consumer group so all N
  // fan-out-receive the same request. Behaviors are applied 1:1 by index;
  // workers are started before returning.
  std::vector<std::unique_ptr<MockKafkaWorker>> makeMigrationWorkerPool(
      std::vector<MockKafkaWorker::Behavior> behaviors) {
    std::vector<std::unique_ptr<MockKafkaWorker>> workers;
    workers.reserve(behaviors.size());
    for (std::size_t i = 0; i < behaviors.size(); ++i) {
      auto worker = std::make_unique<MockKafkaWorker>(
          brokers, requestTopic, ackTopic,
          workerGroup + "-" + std::to_string(i), migrationParser(),
          migrationResponder());
      worker->setBehavior(std::move(behaviors[i]));
      worker->start();
      workers.push_back(std::move(worker));
    }
    return workers;
  }

  // Spin up N migration workers, each pinned via rd_kafka_assign() to
  // partition k on both the request and ack topics. Requires the topics
  // to already have >= behaviors.size() partitions
  // (see ensureMigrationTopicsWithPartitions).
  std::vector<std::unique_ptr<MockKafkaWorker>>
  makePartitionedMigrationWorkerPool(
      std::vector<MockKafkaWorker::Behavior> behaviors) {
    std::vector<std::unique_ptr<MockKafkaWorker>> workers;
    workers.reserve(behaviors.size());
    for (std::size_t i = 0; i < behaviors.size(); ++i) {
      const auto partition = static_cast<int32_t>(i);
      auto worker = std::make_unique<MockKafkaWorker>(
          brokers, requestTopic, ackTopic,
          workerGroup + "-p" + std::to_string(partition), migrationParser(),
          migrationResponder(), partition);
      worker->setBehavior(std::move(behaviors[i]));
      worker->start();
      workers.push_back(std::move(worker));
    }
    return workers;
  }

  static MigrationRequest sampleRequest() {
    return MigrationRequest{.src_slot = 1,
                            .dst_slot = 2,
                            .layer_id = 3,
                            .position_start = 0,
                            .position_end = 128};
  }

  std::string brokers;
  std::string requestTopic;
  std::string ackTopic;
  std::string clientGroup;
  std::string workerGroup;
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

constexpr int32_t K_PARTITIONED_NUM_WORKERS = 4;
constexpr uint32_t K_PARTITIONED_NUM_LAYERS = 64;
constexpr uint32_t K_PARTITIONED_LAYERS_PER_WORKER =
    K_PARTITIONED_NUM_LAYERS / K_PARTITIONED_NUM_WORKERS;
constexpr uint32_t K_PARTITIONED_TARGET_LAYER = 20;
constexpr int32_t K_PARTITIONED_EXPECTED_OWNER = 1;

RemoteKVManagerImpl::LayerToPartition partitionedLayerOwner() {
  return [](uint32_t layerId) -> int32_t {
    return static_cast<int32_t>(layerId / K_PARTITIONED_LAYERS_PER_WORKER);
  };
}

TEST_F(RemoteKVManagerE2ETest, MigrateRoutesRequestToOwnerPartition) {
  ASSERT_EQ(partitionedLayerOwner()(K_PARTITIONED_TARGET_LAYER),
            K_PARTITIONED_EXPECTED_OWNER);
  ASSERT_TRUE(ensureMigrationTopicsWithPartitions(K_PARTITIONED_NUM_WORKERS))
      << "failed to create migration req/ack topics with "
      << K_PARTITIONED_NUM_WORKERS << " partitions";

  auto manager = makeManager(/*timeout=*/30s, partitionedLayerOwner());
  auto workers = makePartitionedMigrationWorkerPool({
      {.dropRequest = true},
      {.replyStatus = MigrationStatus::SUCCESSFUL},
      {.dropRequest = true},
      {.dropRequest = true},
  });
  ASSERT_EQ(workers.size(),
            static_cast<std::size_t>(K_PARTITIONED_NUM_WORKERS));

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  auto req = sampleRequest();
  req.layer_id = K_PARTITIONED_TARGET_LAYER;
  const uint64_t id = manager->migrate(req);
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL;
      },
      POLL_DEADLINE))
      << "routed migration did not reach SUCCESSFUL within "
      << POLL_DEADLINE.count() << "ms";

  EXPECT_EQ(workers[K_PARTITIONED_EXPECTED_OWNER]->requestsReceived(), 1u)
      << "owner worker " << K_PARTITIONED_EXPECTED_OWNER
      << " did not receive the request";
  for (int32_t k = 0; k < K_PARTITIONED_NUM_WORKERS; ++k) {
    if (k == K_PARTITIONED_EXPECTED_OWNER) continue;
    EXPECT_EQ(workers[k]->requestsReceived(), 0u)
        << "non-owner worker " << k << " received a request it does not own";
  }

  const auto raw = workers[K_PARTITIONED_EXPECTED_OWNER]->takeReceivedRaw();
  ASSERT_EQ(raw.size(), 1u);
  const auto parsed = tt::messaging::parseMigrationRequest(raw[0]);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->layer_id, K_PARTITIONED_TARGET_LAYER);
  EXPECT_EQ(parsed->migration_id, id);

  for (auto& w : workers) w->stop();
}

// Owner-partition worker returns FAILED. Manager must observe FAILED via
// the ack (not the sweeper), and no non-owner worker should have seen the
// request.
TEST_F(RemoteKVManagerE2ETest, PartitionedOwnerReplyFailedPropagates) {
  ASSERT_EQ(partitionedLayerOwner()(K_PARTITIONED_TARGET_LAYER),
            K_PARTITIONED_EXPECTED_OWNER);
  ASSERT_TRUE(ensureMigrationTopicsWithPartitions(K_PARTITIONED_NUM_WORKERS));

  // Timeout deliberately longer than POLL_DEADLINE so a FAILED terminal
  // observed here proves the ack path -- not the sweeper -- flipped it.
  auto manager = makeManager(/*timeout=*/30s, partitionedLayerOwner());
  auto workers = makePartitionedMigrationWorkerPool({
      {.dropRequest = true},
      {.replyStatus = MigrationStatus::FAILED},
      {.dropRequest = true},
      {.dropRequest = true},
  });
  ASSERT_EQ(workers.size(),
            static_cast<std::size_t>(K_PARTITIONED_NUM_WORKERS));

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  auto req = sampleRequest();
  req.layer_id = K_PARTITIONED_TARGET_LAYER;
  const uint64_t id = manager->migrate(req);
  ASSERT_NE(id, 0u);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getMigrationStatus(id) == MigrationStatus::FAILED;
      },
      POLL_DEADLINE))
      << "owner-partition FAILED ack did not propagate within "
      << POLL_DEADLINE.count() << "ms";

  EXPECT_EQ(workers[K_PARTITIONED_EXPECTED_OWNER]->requestsReceived(), 1u);
  for (int32_t k = 0; k < K_PARTITIONED_NUM_WORKERS; ++k) {
    if (k == K_PARTITIONED_EXPECTED_OWNER) continue;
    EXPECT_EQ(workers[k]->requestsReceived(), 0u)
        << "non-owner worker " << k << " received a request it does not own";
  }

  for (auto& w : workers) w->stop();
}

TEST_F(RemoteKVManagerE2ETest, PartitionedOwnerDropFailsViaTimeoutSweeper) {
  ASSERT_EQ(partitionedLayerOwner()(K_PARTITIONED_TARGET_LAYER),
            K_PARTITIONED_EXPECTED_OWNER);
  ASSERT_TRUE(ensureMigrationTopicsWithPartitions(K_PARTITIONED_NUM_WORKERS));

  // Short manager timeout so the sweeper resolves the migration well
  // inside POLL_DEADLINE.
  constexpr auto managerTimeout = 1s;
  auto manager = makeManager(managerTimeout, partitionedLayerOwner());
  auto workers = makePartitionedMigrationWorkerPool({
      {.dropRequest = true},
      {.dropRequest = true},
      {.dropRequest = true},
      {.dropRequest = true},
  });
  ASSERT_EQ(workers.size(),
            static_cast<std::size_t>(K_PARTITIONED_NUM_WORKERS));

  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  auto req = sampleRequest();
  req.layer_id = K_PARTITIONED_TARGET_LAYER;
  const uint64_t id = manager->migrate(req);
  ASSERT_NE(id, 0u);
  EXPECT_EQ(manager->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(waitFor(
      [&] {
        return manager->getMigrationStatus(id) == MigrationStatus::FAILED;
      },
      POLL_DEADLINE))
      << "sweeper never marked the dropped partitioned migration FAILED "
      << "within " << POLL_DEADLINE.count() << "ms";

  // Owner consumed the request off its partition, it just chose not to
  // ack (worker crashed mid-flight simulation on a specific partition).
  EXPECT_EQ(workers[K_PARTITIONED_EXPECTED_OWNER]->requestsReceived(), 1u);
  for (int32_t k = 0; k < K_PARTITIONED_NUM_WORKERS; ++k) {
    if (k == K_PARTITIONED_EXPECTED_OWNER) continue;
    EXPECT_EQ(workers[k]->requestsReceived(), 0u)
        << "non-owner worker " << k << " received a request it does not own";
  }

  for (auto& w : workers) w->stop();
}

}  // namespace
}  // namespace tt::services
