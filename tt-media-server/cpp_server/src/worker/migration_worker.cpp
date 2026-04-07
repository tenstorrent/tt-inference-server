// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

/**
 * @file migration_worker.cpp
 * @brief Implementation of MigrationWorker for consuming Kafka offload messages.
 *
 * This worker is designed to run in separate Drogon instances (consumer servers)
 * that listen for session offload signals from the main producer server.
 *
 * Key responsibilities:
 * - Poll Kafka topic for offload messages
 * - Parse JSON messages
 * - Calculate messaging overhead (latency)
 * - Log received signals for monitoring
 */

#include "worker/migration_worker.hpp"

#include <chrono>
#include <nlohmann/json.hpp>

#include "utils/logger.hpp"

namespace tt::worker {

MigrationWorker::MigrationWorker(MigrationWorkerConfig config)
    : config_(std::move(config)) {
  // Create Kafka consumer with the provided configuration
  consumer_ = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{
          .brokers = config_.brokers,
          .topic = config_.topic,
          .group_id = config_.group_id
      });

  TT_LOG_INFO("[MigrationWorker] Initialized with brokers={}, topic={}, group={}",
              config_.brokers, config_.topic, config_.group_id);
}

MigrationWorker::~MigrationWorker() {
  stop();
}

void MigrationWorker::start() {
  // Check if already running (CAS: compare-and-swap)
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    TT_LOG_WARN("[MigrationWorker] Already running, ignoring start() call");
    return;
  }

  // Launch worker thread
  workerThread_ = std::thread([this] { consumerLoop(); });
  TT_LOG_INFO("[MigrationWorker] Started consumer loop thread");
}

void MigrationWorker::stop() {
  // Check if already stopped
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;  // Already stopped
  }

  // Wait for worker thread to finish
  if (workerThread_.joinable()) {
    workerThread_.join();
    TT_LOG_INFO("[MigrationWorker] Worker thread joined");
  }
}

void MigrationWorker::consumerLoop() {
  TT_LOG_INFO("[MigrationWorker] Entering consumer loop");

  // Poll loop: runs until stopped
  while (running_.load(std::memory_order_relaxed)) {
    // Poll with 1 second timeout
    // Note: Must poll regularly to maintain consumer group membership
    auto msg = consumer_->poll_payload(1000);

    if (msg.has_value()) {
      // Message received - process it
  // Capture receive timestamp immediately for accurate latency measurementm

      auto receiveTime = std::chrono::system_clock::now();

      processOffloadRequest(*msg, receiveTime);
    }
    // If no message (timeout or error), loop continues
  }

  TT_LOG_INFO("[MigrationWorker] Exited consumer loop");
}

void MigrationWorker::processOffloadRequest(const std::string& message, auto receiveTime) {
  auto receiveUs = std::chrono::duration_cast<std::chrono::microseconds>(
      receiveTime.time_since_epoch()).count();

  try {
    // Parse JSON message
    auto json = nlohmann::json::parse(message);

    // Extract timestamp from producer
    int64_t sentUs = json["timestamp_us"].get<int64_t>();

    // Calculate overhead (latency)
    int64_t overheadUs = receiveUs - sentUs;
    double overheadMs = overheadUs / 1000.0;

    // Extract additional fields (optional, for logging)
    std::string action = json.value("action", "unknown");
    int currentCount = json.value("current_session_count", 0);
    int maxSessions = json.value("max_sessions", 0);

    // Log the offload request with overhead measurement
    TT_LOG_WARN("[MigrationWorker] ✅ OFFLOAD REQUEST RECEIVED");
    TT_LOG_WARN("[MigrationWorker]   Action:      {}", action);
    TT_LOG_WARN("[MigrationWorker]   Sessions:    {}/{} ({:.1f}%)",
                currentCount, maxSessions,
                maxSessions > 0 ? (currentCount * 100.0 / maxSessions) : 0.0);
    TT_LOG_WARN("[MigrationWorker]   Sent at:     {} μs", sentUs);
    TT_LOG_WARN("[MigrationWorker]   Received at: {} μs", receiveUs);
    TT_LOG_WARN("[MigrationWorker]   ⏱️  OVERHEAD:  {} μs ({:.3f} ms)",
                overheadUs, overheadMs);
    TT_LOG_WARN("[MigrationWorker]   Raw payload: {}", message);

  // TODO LJUBICA: actual implementation
  // 1. Parse session properly
  // 2. Allocate memory slot
  // 3. Load KV cache
  // etc. 

  } catch (const nlohmann::json::exception& e) {
    TT_LOG_ERROR("[MigrationWorker] Failed to parse JSON message: {}", e.what());
    TT_LOG_ERROR("[MigrationWorker] Invalid message: {}", message);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[MigrationWorker] Error processing message: {}", e.what());
  }
}

}