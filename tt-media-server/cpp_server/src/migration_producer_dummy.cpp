// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Dummy migration producer: publishes fake offload requests to Kafka at a
// configurable rate so that MigrationWorker consumers can be exercised without
// a real inference server running.
//
// Usage:
//   ./build/migration_producer_dummy [options]
//
// Env vars (same as main server):
//   KAFKA_BROKERS             (default: localhost:9092)
//   KAFKA_OFFLOAD_TOPIC_NAME  (default: offload_requests)
//
// CLI flags:
//   --count N      Number of messages to publish (default: 10, 0 = infinite)
//   --interval-ms  Delay between messages in milliseconds (default: 500)
//   --brokers      Override KAFKA_BROKERS
//   --topic        Override KAFKA_OFFLOAD_TOPIC_NAME

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "config/settings.hpp"
#include "messaging/kafka_producer.hpp"
#include "utils/logger.hpp"

namespace {

std::string generateSessionId() {
  static std::mt19937_64 rng(std::random_device{}());
  static const char kHex[] = "0123456789abcdef";
  // Simplified UUID-like string: 8-4-4-4-12
  std::string uuid;
  uuid.reserve(36);
  auto appendHex = [&](int n) {
    for (int i = 0; i < n; ++i) {
      uuid += kHex[rng() % 16];
    }
  };
  appendHex(8);
  uuid += '-';
  appendHex(4);
  uuid += '-';
  appendHex(4);
  uuid += '-';
  appendHex(4);
  uuid += '-';
  appendHex(12);
  return uuid;
}

uint64_t generateMigrationId() {
  static std::mt19937_64 rng(std::random_device{}());
  return rng();
}

int64_t nowMicroseconds() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string buildMessage(int sequenceNumber, const std::string& sourceHost,
                         uint16_t sourcePort, uint64_t sourceOffset,
                         uint64_t kvSize) {
  auto sessionId = generateSessionId();
  auto migrationId = generateMigrationId();
  auto timestampUs = nowMicroseconds();

  // JSON payload matching the schema MigrationWorker expects + extension fields
  // for transport metadata.
  std::string json = "{";
  json += "\"action\":\"migrate_out\",";
  json += "\"session_id\":\"" + sessionId + "\",";
  json += "\"migration_id\":" + std::to_string(migrationId) + ",";
  json += "\"source_host\":\"" + sourceHost + "\",";
  json += "\"source_port\":" + std::to_string(sourcePort) + ",";
  json += "\"source_segment_addr\":" + std::to_string(sourceOffset) + ",";
  json += "\"kv_size_bytes\":" + std::to_string(kvSize) + ",";
  json += "\"source_slot_id\":" + std::to_string(sequenceNumber % 32) + ",";
  json += "\"num_cached_tokens\":" + std::to_string(128 + sequenceNumber * 32) +
          ",";
  json += "\"current_session_count\":" + std::to_string(850 + sequenceNumber) +
          ",";
  json += "\"max_sessions\":1000,";
  json += "\"timestamp_us\":" + std::to_string(timestampUs);
  json += "}";
  return json;
}

}  // namespace

int main(int argc, char* argv[]) {
  int count = 10;
  int intervalMs = 500;
  std::string brokers;
  std::string topic;
  std::string sourceHost = "127.0.0.1";
  uint16_t sourcePort = 17777;
  uint64_t sourceOffset = 0;
  uint64_t kvSize = 65536;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--count" && i + 1 < argc) {
      count = std::stoi(argv[++i]);
    } else if (arg == "--interval-ms" && i + 1 < argc) {
      intervalMs = std::stoi(argv[++i]);
    } else if (arg == "--brokers" && i + 1 < argc) {
      brokers = argv[++i];
    } else if (arg == "--topic" && i + 1 < argc) {
      topic = argv[++i];
    } else if (arg == "--source-host" && i + 1 < argc) {
      sourceHost = argv[++i];
    } else if (arg == "--source-port" && i + 1 < argc) {
      sourcePort = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if (arg == "--source-offset" && i + 1 < argc) {
      sourceOffset = std::stoull(argv[++i]);
    } else if (arg == "--kv-size" && i + 1 < argc) {
      kvSize = std::stoull(argv[++i]);
    } else if (arg == "--help") {
      std::cout
          << "Migration Producer Dummy\n"
          << "Publishes fake offload requests to Kafka for testing.\n\n"
          << "Usage: " << argv[0] << " [options]\n"
          << "Options:\n"
          << "  --count N          Messages to send (0=infinite, default: 10)\n"
          << "  --interval-ms MS   Delay between messages (default: 500)\n"
          << "  --brokers ADDRS    Kafka brokers (default: env or "
             "localhost:9092)\n"
          << "  --topic NAME       Kafka topic (default: env or "
             "offload_requests)\n"
          << "  --source-host H    Source segment host (default: 127.0.0.1)\n"
          << "  --source-port P    Source segment port (default: 17777)\n"
          << "  --source-offset O  Offset within source buffer (default: 0)\n"
          << "  --kv-size BYTES    KV cache size to transfer (default: "
             "65536)\n"
          << "  --help             Show this help\n";
      return 0;
    }
  }

  tt::utils::ZeroOverheadLogger::initialize("producer");

  if (brokers.empty()) brokers = tt::config::kafkaBrokers();
  if (topic.empty()) topic = tt::config::kafkaOffloadTopicName();

  TT_LOG_INFO("[Producer] brokers={} topic={} count={} interval={}ms", brokers,
              topic, count, intervalMs);

  tt::messaging::KafkaProducer producer(
      tt::messaging::KafkaProducerConfig{.brokers = brokers, .topic = topic});

  int sent = 0;
  const bool infinite = (count == 0);

  while (infinite || sent < count) {
    auto message = buildMessage(sent, sourceHost, sourcePort, sourceOffset, kvSize);
    std::string error;
    if (producer.send(message, &error)) {
      TT_LOG_INFO("[Producer] Sent message {} : {}", sent, message);
      ++sent;
    } else {
      TT_LOG_ERROR("[Producer] Failed to send message {}: {}", sent, error);
    }

    if (intervalMs > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
    }
  }

  TT_LOG_INFO("[Producer] Flushing...");
  std::string flushErr;
  if (!producer.flush(5000, &flushErr)) {
    TT_LOG_ERROR("[Producer] Flush failed: {}", flushErr);
  }
  TT_LOG_INFO("[Producer] Done. Sent {} messages.", sent);
  return 0;
}
