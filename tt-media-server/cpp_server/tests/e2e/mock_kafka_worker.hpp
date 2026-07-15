// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Real-Kafka-backed drop-in for tt::worker::KvMigrationWorker whose reply
// behavior is scriptable at runtime. Path-specific parsing and response
// building are supplied at construction time as callables so that follow-up
// message families (download, offload, ...) can reuse this one class rather
// than growing near-identical copies.
//
// Received request payloads are kept as raw JSON in an internal log so
// tests that need to assert on wire content can parse them themselves.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "messaging/migration_message.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::services::testing {

class MockKafkaWorker {
 public:
  struct Behavior {
    MigrationStatus replyStatus{MigrationStatus::SUCCESSFUL};
    std::chrono::milliseconds replyDelay{0};
    bool dropRequest{false};
  };

  // Parse a raw request payload. Return the request's id on success, nullopt
  // on parse failure (the message is dropped without acking).
  using RequestParser =
      std::function<std::optional<uint64_t>(const std::string& raw)>;
  // Build the response payload for a given (id, behavior) pair.
  using ResponseBuilder =
      std::function<std::string(uint64_t id, const Behavior& behavior)>;

  MockKafkaWorker(const std::string& brokers, const std::string& requestTopic,
                  const std::string& ackTopic, const std::string& groupId,
                  RequestParser parser, ResponseBuilder responder,
                  std::optional<int32_t> partition = std::nullopt)
      : parseRequest{std::move(parser)},
        buildResponse{std::move(responder)},
        producePartition{partition} {
    requestConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
        tt::messaging::KafkaConsumerConfig{.brokers = brokers,
                                           .topic = requestTopic,
                                           .group_id = groupId,
                                           .partition = partition});
    ackProducer = std::make_unique<tt::messaging::KafkaProducer>(
        tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                           .topic = ackTopic});
  }

  ~MockKafkaWorker() { stop(); }

  MockKafkaWorker(const MockKafkaWorker&) = delete;
  MockKafkaWorker& operator=(const MockKafkaWorker&) = delete;

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

  std::vector<std::string> takeReceivedRaw() {
    std::lock_guard<std::mutex> lock(mtx);
    auto out = std::move(receivedRaw);
    receivedRaw.clear();
    return out;
  }

 private:
  void run() {
    while (running.load(std::memory_order_relaxed)) {
      auto raw = requestConsumer->receive(50);
      if (!raw.has_value()) continue;

      auto id = parseRequest(*raw);
      if (!id.has_value()) continue;

      received.fetch_add(1, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(mtx);
        receivedRaw.push_back(*raw);
      }

      const Behavior b = getBehavior();
      if (b.dropRequest) continue;

      if (b.replyDelay.count() > 0) {
        std::this_thread::sleep_for(b.replyDelay);
      }

      const std::string payload = buildResponse(*id, b);
      std::string err;
      if (producePartition.has_value()) {
        ackProducer->send(payload, *producePartition, &err);
      } else {
        ackProducer->send(payload, &err);
      }
    }
  }

  RequestParser parseRequest;
  ResponseBuilder buildResponse;
  std::optional<int32_t> producePartition;
  std::unique_ptr<tt::messaging::KafkaConsumer> requestConsumer;
  std::unique_ptr<tt::messaging::KafkaProducer> ackProducer;
  std::atomic<bool> running{false};
  std::thread thread;
  mutable std::mutex mtx;
  Behavior behavior;
  std::vector<std::string> receivedRaw;
  std::atomic<std::size_t> received{0};
};

// ---------------------------------------------------------------------------
// Path-specific parser / responder factories
// ---------------------------------------------------------------------------

inline MockKafkaWorker::RequestParser migrationParser() {
  return [](const std::string& raw) -> std::optional<uint64_t> {
    auto parsed = tt::messaging::parseMigrationRequest(raw);
    if (!parsed.has_value()) return std::nullopt;
    return parsed->migration_id;
  };
}

inline MockKafkaWorker::ResponseBuilder migrationResponder() {
  return [](uint64_t id, const MockKafkaWorker::Behavior& b) {
    return tt::messaging::serialize(tt::messaging::MigrationResponseMessage{
        .migration_id = id, .status = b.replyStatus});
  };
}

}  // namespace tt::services::testing
