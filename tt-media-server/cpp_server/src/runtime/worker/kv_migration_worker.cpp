// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/worker/kv_migration_worker.hpp"

#include <utility>

#include "messaging/migration_message.hpp"
#include "utils/logger.hpp"

namespace tt::worker {

KvMigrationWorker::KvMigrationWorker(
    std::unique_ptr<tt::messaging::IKafkaConsumer> requestConsumer,
    std::unique_ptr<tt::messaging::IKafkaProducer> ackProducer,
    std::unique_ptr<IMigrationExecutor> executor, int pollTimeoutMs)
    : requestConsumer(std::move(requestConsumer)),
      ackProducer(std::move(ackProducer)),
      executor(std::move(executor)),
      pollTimeoutMs(pollTimeoutMs) {
  if (!this->requestConsumer) {
    TT_LOG_ERROR(
        "[KvMigrationWorker] null requestConsumer; start() will spin idle");
  }
  if (!this->ackProducer) {
    TT_LOG_ERROR("[KvMigrationWorker] null ackProducer; acks will be dropped");
  }
  if (!this->executor) {
    TT_LOG_ERROR(
        "[KvMigrationWorker] null executor; requests will be parsed but never "
        "executed");
  }
}

KvMigrationWorker::~KvMigrationWorker() { stop(); }

void KvMigrationWorker::start() {
  bool expected = false;
  if (!running.compare_exchange_strong(expected, true)) {
    TT_LOG_WARN("[KvMigrationWorker] already running; ignoring start()");
    return;
  }
  workerThread = std::thread([this] { consumerLoop(); });
  TT_LOG_INFO("[KvMigrationWorker] started (poll={}ms)", pollTimeoutMs);
}

void KvMigrationWorker::stop() {
  bool expected = true;
  if (!running.compare_exchange_strong(expected, false)) {
    return;
  }
  if (workerThread.joinable()) {
    workerThread.join();
  }
  TT_LOG_INFO("[KvMigrationWorker] stopped");
}

void KvMigrationWorker::consumerLoop() {
  TT_LOG_INFO("[KvMigrationWorker] consumer loop entered");

  while (running.load(std::memory_order_relaxed)) {
    if (!requestConsumer) {
      std::this_thread::sleep_for(std::chrono::milliseconds(pollTimeoutMs));
      continue;
    }

    auto raw = requestConsumer->receive(pollTimeoutMs);
    if (!raw.has_value()) {
      continue;
    }

    auto parsed = tt::messaging::parseMigrationRequest(*raw);
    if (!parsed.has_value()) {
      TT_LOG_WARN("[KvMigrationWorker] dropping unparseable request: {}", *raw);
      continue;
    }

    const uint64_t migrationId = parsed->migration_id;
    const tt::services::MigrationRequest apiReq{
        .src_slot = parsed->src_slot,
        .dst_slot = parsed->dst_slot,
        .layer_begin = parsed->layer_begin,
        .layer_end = parsed->layer_end,
        .src_position_begin = parsed->src_position_begin,
        .src_position_end = parsed->src_position_end,
        .dst_position_begin = parsed->dst_position_begin,
        .dst_position_end = parsed->dst_position_end,
    };

    TT_LOG_DEBUG("[KvMigrationWorker] dispatching migration_id={} to executor",
                 migrationId);

    if (!executor) {
      // Surface the failure rather than silently dropping the request.
      publishAck(migrationId, tt::services::MigrationStatus::FAILED);
      continue;
    }

    // execute() is contractually non-blocking; the callback may fire on
    // this thread (synchronous Stub) or on an executor-owned thread.
    executor->execute(
        migrationId, apiReq,
        [this, migrationId](tt::services::MigrationStatus status) {
          publishAck(migrationId, status);
        });
  }

  TT_LOG_INFO("[KvMigrationWorker] consumer loop exited");
}

void KvMigrationWorker::publishAck(uint64_t migrationId,
                                   tt::services::MigrationStatus status) {
  const tt::messaging::MigrationResponseMessage ackMsg{
      .migration_id = migrationId,
      .status = status,
  };
  const std::string payload = tt::messaging::serialize(ackMsg);

  if (!ackProducer) {
    TT_LOG_ERROR(
        "[KvMigrationWorker] no ackProducer; cannot publish ack for "
        "migration_id={}",
        migrationId);
    return;
  }

  std::string err;
  bool sent = false;
  {
    // KafkaProducer::send is thread-safe at the librdkafka layer, but we
    // also want to serialize against any future producer-state mutation.
    std::lock_guard<std::mutex> lock(ackMutex);
    sent = ackProducer->send(payload, &err);
  }

  if (!sent) {
    TT_LOG_ERROR(
        "[KvMigrationWorker] ackProducer.send failed for migration_id={}: {}",
        migrationId, err);
  } else {
    TT_LOG_DEBUG("[KvMigrationWorker] published ack migration_id={} status={}",
                 migrationId, static_cast<int>(status));
  }
}

}  // namespace tt::worker
