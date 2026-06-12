// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/worker/migration_worker.hpp"

#include <json/json.h>

#include <chrono>
#include <sstream>
#include <utility>

#include "transport/transfer_types.hpp"
#include "utils/logger.hpp"

namespace tt::worker {

MigrationWorker::MigrationWorker(
    MigrationWorkerConfig config,
    std::shared_ptr<tt::transport::ITransferEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {
  consumer_ = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{
          .brokers = config_.brokers,
          .topic = config_.topic,
          .group_id = config_.group_id,
      });

  // Pre-allocate the receive buffer (touches all pages now, not during transfer).
  recvBuffer_.resize(config_.max_transfer_size, 0);

  TT_LOG_INFO(
      "[MigrationWorker] Initialized with brokers={}, topic={}, group={}, "
      "transferEngine={}, recvBuffer={}MB",
      config_.brokers, config_.topic, config_.group_id,
      engine_ ? "enabled" : "disabled",
      config_.max_transfer_size / (1024 * 1024));
}

MigrationWorker::~MigrationWorker() { stop(); }

void MigrationWorker::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    TT_LOG_WARN("[MigrationWorker] Already running, ignoring start() call");
    return;
  }

  // Register pre-allocated buffer with the transfer engine once.
  if (engine_ && !bufferRegistered_) {
    if (engine_->registerLocalMemory(recvBuffer_.data(), recvBuffer_.size())) {
      bufferRegistered_ = true;
      TT_LOG_INFO("[MigrationWorker] Pre-registered {}MB receive buffer",
                  recvBuffer_.size() / (1024 * 1024));
    } else {
      TT_LOG_ERROR(
          "[MigrationWorker] Failed to register receive buffer — transfers "
          "will allocate per-call");
    }
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

  // Unregister the pre-allocated buffer.
  if (engine_ && bufferRegistered_) {
    engine_->unregisterLocalMemory(recvBuffer_.data());
    bufferRegistered_ = false;
    TT_LOG_INFO("[MigrationWorker] Unregistered receive buffer");
  }
}

void MigrationWorker::consumerLoop() {
  TT_LOG_INFO("[MigrationWorker] Entering consumer loop");

  while (running_.load(std::memory_order_relaxed)) {
    auto msg = consumer_->receive(100);

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

  auto req = parseRequest(message);
  if (req.timestampUs == 0) {
    return;  // parse failed, already logged
  }

  const Json::Int64 overheadUs = receiveUs - req.timestampUs;
  const double overheadMs = static_cast<double>(overheadUs) / 1000.0;

  TT_LOG_INFO("[MigrationWorker] ✅ OFFLOAD REQUEST RECEIVED");
  TT_LOG_INFO("[MigrationWorker]   Action:      {}", req.action);
  TT_LOG_INFO("[MigrationWorker]   Session ID:  {}", req.sessionId);
  TT_LOG_INFO("[MigrationWorker]   Migration:   {}", req.migrationId);
  TT_LOG_INFO("[MigrationWorker]   Source:      {}:{} slot={} tokens={}",
              req.sourceHost, req.sourcePort, req.sourceSlotId,
              req.numCachedTokens);
  TT_LOG_INFO("[MigrationWorker]   KV size:     {} bytes", req.kvSizeBytes);
  TT_LOG_INFO("[MigrationWorker]   ⏱️  OVERHEAD:  {} μs ({:.3f} ms)",
              overheadUs, overheadMs);

  executeMigration(req);
}

MigrationRequest MigrationWorker::parseRequest(const std::string& message) {
  MigrationRequest req;

  Json::Value root;
  Json::CharReaderBuilder builder;
  std::istringstream iss(message);
  std::string parseErrors;
  if (!Json::parseFromStream(builder, iss, &root, &parseErrors)) {
    TT_LOG_ERROR("[MigrationWorker] JSON parse failed: {}", parseErrors);
    return req;
  }

  if (!root.isMember("timestamp_us") || !root["timestamp_us"].isIntegral()) {
    TT_LOG_ERROR("[MigrationWorker] Missing or non-integral timestamp_us");
    return req;
  }

  req.action = root.get("action", "unknown").asString();
  req.sessionId = root.get("session_id", "").asString();
  req.migrationId = root.get("migration_id", 0).asUInt64();
  req.sourceHost = root.get("source_host", "").asString();
  req.sourcePort =
      static_cast<uint16_t>(root.get("source_port", 0).asUInt());
  req.sourceSegmentAddr = root.get("source_segment_addr", 0).asUInt64();
  req.kvSizeBytes = root.get("kv_size_bytes", 0).asUInt64();
  req.sourceSlotId = root.get("source_slot_id", 0).asUInt();
  req.numCachedTokens = root.get("num_cached_tokens", 0).asUInt();
  req.timestampUs = root["timestamp_us"].asInt64();

  return req;
}

void MigrationWorker::executeMigration(const MigrationRequest& req) {
  if (!engine_) {
    TT_LOG_INFO(
        "[MigrationWorker] No transfer engine configured; skipping RDMA "
        "transfer for migrationId={}",
        req.migrationId);
    return;
  }

  if (req.kvSizeBytes == 0) {
    TT_LOG_WARN("[MigrationWorker] kvSizeBytes=0, nothing to transfer");
    return;
  }

  if (req.kvSizeBytes > recvBuffer_.size()) {
    TT_LOG_ERROR(
        "[MigrationWorker] Transfer size {} exceeds pre-allocated buffer "
        "{}MB for migrationId={}",
        req.kvSizeBytes, recvBuffer_.size() / (1024 * 1024), req.migrationId);
    return;
  }

  // Build the peer segment name from source host:port.
  std::string peerSegment =
      req.sourceHost + ":" + std::to_string(req.sourcePort);

  TT_LOG_INFO(
      "[MigrationWorker] Starting RDMA pull: migrationId={} peer={} "
      "offset={:#x} size={}",
      req.migrationId, peerSegment, req.sourceSegmentAddr, req.kvSizeBytes);

  auto t0 = std::chrono::steady_clock::now();

  // Open the remote segment (cached after first call for same peer).
  auto segment = engine_->openSegment(peerSegment);
  if (segment == tt::transport::kInvalidSegment) {
    TT_LOG_ERROR(
        "[MigrationWorker] Failed to open segment '{}' for migrationId={}",
        peerSegment, req.migrationId);
    return;
  }

  auto t1 = std::chrono::steady_clock::now();

  // Pull from remote segment into the pre-allocated, pre-registered buffer.
  tt::transport::TransferRequest xfer;
  xfer.op = tt::transport::TransferOp::Read;
  xfer.local_addr = recvBuffer_.data();
  xfer.target = segment;
  xfer.target_offset = req.sourceSegmentAddr;
  xfer.length = req.kvSizeBytes;

  auto status = engine_->submitAndWait(xfer);

  auto t2 = std::chrono::steady_clock::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  TT_LOG_INFO(
      "[MigrationWorker] ⏱️  BREAKDOWN: openSegment={:.1f}ms "
      "transfer={:.1f}ms TOTAL={:.1f}ms",
      ms(t0, t1), ms(t1, t2), ms(t0, t2));

  if (status.state == tt::transport::TransferState::Completed) {
    TT_LOG_INFO(
        "[MigrationWorker] ✅ RDMA pull complete: migrationId={} "
        "transferred={} bytes",
        req.migrationId, status.transferred_bytes);
  } else {
    TT_LOG_ERROR(
        "[MigrationWorker] ❌ RDMA pull failed: migrationId={} state={}",
        req.migrationId, static_cast<int>(status.state));
  }
}

}  // namespace tt::worker
