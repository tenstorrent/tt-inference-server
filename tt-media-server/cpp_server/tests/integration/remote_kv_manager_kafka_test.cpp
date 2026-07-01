// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Phase 4 integration test: real Kafka broker, real RemoteKVManagerImpl, real
// KvMigrationWorker, real KafkaProducer / KafkaConsumer, real serialization.
// The only thing still mocked is the byte movement itself, which goes through
// StubMigrationExecutor (always SUCCESSFUL, except where the test wires it
// otherwise).
//
// Opt-in: this test exits with GTEST_SKIP unless INTEGRATION_TESTS_ENABLED=1
// is set, so a developer running `ctest` without a broker handy does not see
// a spurious failure. CI is expected to set the env var.
//
// Broker address: KAFKA_BROKERS env var, default "localhost:9092".
//
// Each test mints fresh topic names and consumer-group ids derived from the
// pid + microsecond timestamp, so concurrent runs (and the dev
// tt_kv_migration_consumer binary that may be subscribed to the production
// topics) cannot leak into the test.

#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "runtime/worker/kv_migration_worker.hpp"
#include "runtime/worker/stub_migration_executor.hpp"
#include "services/remote_kv_manager.hpp"
#include "services/remote_kv_manager_impl.hpp"

namespace tt::services {
namespace {

using namespace std::chrono_literals;

// A real Kafka consumer can take ~1-2 s to complete partition assignment
// after subscribing. We must NOT issue any migrate() (which produces) until
// both the worker's request-consumer and the manager's ack-consumer have
// joined their groups, otherwise auto.offset.reset=latest causes them to
// miss messages produced before they were ready. 3 s is a conservative
// margin observed to be reliable on local broker setups.
constexpr auto KAFKA_GROUP_JOIN_WARMUP = 3s;

// Generous because every step touches a real broker.
constexpr auto COMPLETION_TIMEOUT = 15s;

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

class RemoteKVManagerKafkaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (std::getenv("INTEGRATION_TESTS_ENABLED") == nullptr) {
      GTEST_SKIP() << "Set INTEGRATION_TESTS_ENABLED=1 (and ensure a Kafka "
                      "broker is reachable at KAFKA_BROKERS) to run.";
    }

    brokers = envOr("KAFKA_BROKERS", "localhost:9092");

    const auto suffix = uniqueSuffix();
    requestTopic = "it-kv-req-" + suffix;
    ackTopic = "it-kv-ack-" + suffix;
    clientGroup = "it-client-" + suffix;
    workerGroup = "it-worker-" + suffix;
  }

  std::unique_ptr<RemoteKVManagerImpl> makeManager() {
    auto producer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{
            .brokers = brokers,
            .topic = requestTopic,
        });
    auto consumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{
            .brokers = brokers,
            .topic = ackTopic,
            .group_id = clientGroup,
        });
    return std::make_unique<RemoteKVManagerImpl>(
        std::move(producer), std::move(consumer),
        /*migrationWorkerPoolSize=*/1, /*timeout=*/30s,
        /*sweepInterval=*/200ms, /*drainPollMs=*/50);
  }

  std::unique_ptr<tt::worker::KvMigrationWorker> makeWorker(
      MigrationStatus stubResult = MigrationStatus::SUCCESSFUL) {
    auto consumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{
            .brokers = brokers,
            .topic = requestTopic,
            .group_id = workerGroup,
        });
    auto producer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{
            .brokers = brokers,
            .topic = ackTopic,
        });
    auto executor =
        std::make_unique<tt::worker::StubMigrationExecutor>(stubResult);
    return std::make_unique<tt::worker::KvMigrationWorker>(
        std::move(consumer), std::move(producer), std::move(executor),
        /*pollTimeoutMs=*/50);
  }

  static MigrationRequest sampleRequest(uint32_t src = 1, uint32_t dst = 2) {
    return MigrationRequest{
        .src_slot = src,
        .dst_slot = dst,
        .layer_id = 3,
        .position_start = 0,
        .position_end = 128,
    };
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

TEST_F(RemoteKVManagerKafkaTest, MigrateRoundTripCompletes) {
  auto manager = makeManager();
  auto worker = makeWorker();
  worker->start();

  // Wait for both consumer groups to join.
  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->migrate(sampleRequest(7, 9));
  EXPECT_NE(id, 0u);
  EXPECT_EQ(manager->getMigrationStatus(id), MigrationStatus::IN_PROGRESS);

  ASSERT_TRUE(waitFor(
      [&] { return manager->getMigrationStatus(id) == MigrationStatus::SUCCESSFUL; },
      COMPLETION_TIMEOUT))
      << "migration did not complete within " << COMPLETION_TIMEOUT.count()
      << "ms; check broker, topic auto-creation, and consumer-group warmup";

  // Order matters: stop the worker first so its in-flight callbacks finish
  // before the manager (and its ack consumer) goes away.
  worker->stop();
}

TEST_F(RemoteKVManagerKafkaTest, MultipleMigrationsAllComplete) {
  auto manager = makeManager();
  auto worker = makeWorker();
  worker->start();
  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  constexpr int migrations = 5;
  std::vector<uint64_t> ids;
  ids.reserve(migrations);
  for (int i = 0; i < migrations; ++i) {
    ids.push_back(manager->migrate(sampleRequest(
        /*src=*/static_cast<uint32_t>(i),
        /*dst=*/static_cast<uint32_t>(i + 1))));
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
      COMPLETION_TIMEOUT))
      << "not every migration reached SUCCESSFUL within "
      << COMPLETION_TIMEOUT.count() << "ms";

  worker->stop();
}

TEST_F(RemoteKVManagerKafkaTest, ConfiguredFailureStatusPropagates) {
  auto manager = makeManager();
  auto worker = makeWorker(/*stubResult=*/MigrationStatus::FAILED);
  worker->start();
  std::this_thread::sleep_for(KAFKA_GROUP_JOIN_WARMUP);

  const uint64_t id = manager->migrate(sampleRequest());

  ASSERT_TRUE(
      waitFor([&] { return manager->getMigrationStatus(id) == MigrationStatus::FAILED; },
              COMPLETION_TIMEOUT))
      << "FAILED ack did not arrive within " << COMPLETION_TIMEOUT.count()
      << "ms";

  worker->stop();
}

TEST_F(RemoteKVManagerKafkaTest, UnknownIdReturnsUnknown) {
  auto manager = makeManager();
  // No worker needed; we never publish anything for this id.
  EXPECT_EQ(manager->getMigrationStatus(/*never-issued=*/0xCAFEBABEDEADBEEFull),
            MigrationStatus::UNKNOWN);
}

}  // namespace
}  // namespace tt::services
