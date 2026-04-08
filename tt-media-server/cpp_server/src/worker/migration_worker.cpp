// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/migration_worker.hpp"

#include <chrono>
#include <json/json.h>
#include <sstream>

#include "utils/logger.hpp"

namespace tt::worker {

MigrationWorker::MigrationWorker(MigrationWorkerConfig config)
    : config_(std::move(config)) {
  consumer_ = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{
          .brokers = config_.brokers,
          .topic = config_.topic,
          .group_id = config_.group_id,
      });

  TT_LOG_INFO("[MigrationWorker] Initialized with brokers={}, topic={}, group={}",
              config_.brokers, config_.topic, config_.group_id);
}

MigrationWorker::~MigrationWorker() { stop(); }

void MigrationWorker::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    TT_LOG_WARN("[MigrationWorker] Already running, ignoring start() call");
    return;
  }
  workerThread_ = std::thread([this] { consumerLoop(); });
  TT_LOG_INFO("[MigrationWorker] Started consumer loop thread");
}

void MigrationWorker::stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  if (workerThread_.joinable()) {
    workerThread_.join();
    TT_LOG_INFO("[MigrationWorker] Worker thread joined");
  }
}

void MigrationWorker::consumerLoop() {
  TT_LOG_INFO("[MigrationWorker] Entering consumer loop");

  while (running_.load(std::memory_order_relaxed)) {
    auto msg = consumer_->pollPayload(1);

    if (msg.has_value()) {
      auto receiveTime = std::chrono::system_clock::now();
      processOffloadRequest(*msg, receiveTime);
    }
  }

  TT_LOG_INFO("[MigrationWorker] Exited consumer loop");
}

void MigrationWorker::processOffloadRequest(
    const std::string& message,
    std::chrono::system_clock::time_point receiveTime) {
  const auto receiveUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             receiveTime.time_since_epoch())
                             .count();

  Json::Value root;
  Json::CharReaderBuilder builder;
  std::istringstream iss(message);
  std::string parseErrors;
  if (!Json::parseFromStream(builder, iss, &root, &parseErrors)) {
    TT_LOG_ERROR("[MigrationWorker] JSON parse failed: {}", parseErrors);
    TT_LOG_ERROR("[MigrationWorker] Invalid message: {}", message);
    return;
  }

  if (!root.isMember("timestamp_us") || !root["timestamp_us"].isIntegral()) {
    TT_LOG_ERROR("[MigrationWorker] Missing or non-integral timestamp_us");
    TT_LOG_ERROR("[MigrationWorker] Invalid message: {}", message);
    return;
  }

  const Json::Int64 sentUs = root["timestamp_us"].asInt64();
  const Json::Int64 overheadUs = receiveUs - sentUs;
  const double overheadMs = static_cast<double>(overheadUs) / 1000.0;

  const std::string action = root.get("action", Json::Value("unknown")).asString();
  const std::string sessionId = root.get("session_id", Json::Value("unknown")).asString();
  const int currentCount = root.get("current_session_count", Json::Value(0)).asInt();
  const int maxSessions = root.get("max_sessions", Json::Value(0)).asInt();

  TT_LOG_WARN("[MigrationWorker] ✅ OFFLOAD REQUEST RECEIVED");
  TT_LOG_WARN("[MigrationWorker]   Action:      {}", action);
  TT_LOG_WARN("[MigrationWorker]   Session ID:  {}", sessionId);
  TT_LOG_WARN("[MigrationWorker]   Sessions:    {}/{} ({:.1f}%)",
              currentCount, maxSessions,
              maxSessions > 0 ? (currentCount * 100.0 / maxSessions) : 0.0);
  TT_LOG_WARN("[MigrationWorker]   Sent at:     {} μs", sentUs);
  TT_LOG_WARN("[MigrationWorker]   Received at: {} μs", receiveUs);
  TT_LOG_WARN("[MigrationWorker]   ⏱️  OVERHEAD:  {} μs ({:.3f} ms)", overheadUs, overheadMs);
  TT_LOG_WARN("[MigrationWorker]   Raw payload: {}", message);
}

}  // namespace tt::worker
