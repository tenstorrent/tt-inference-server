// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/worker/kv_migration_worker.hpp"

#include <string>
#include <utility>

#include "messaging/migration_message.hpp"
#include "utils/logger.hpp"

namespace tt::worker {

KvMigrationWorker::KvMigrationWorker(
    std::unique_ptr<tt::messaging::IKafkaConsumer> requestConsumer,
    std::unique_ptr<tt::messaging::IKafkaProducer> ackProducer,
    std::unique_ptr<IMigrationExecutor> executor, int pollTimeoutMs,
    std::optional<int32_t> ackPartition)
    : requestConsumer(std::move(requestConsumer)),
      ackProducer(std::move(ackProducer)),
      executor(std::move(executor)),
      pollTimeoutMs(pollTimeoutMs),
      ackPartition(ackPartition) {
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

KvMigrationWorker::~KvMigrationWorker() {
  stop();  // join the poll thread first: no new jobs are submitted after this.
  // Then drain the executor while this worker is still fully valid. An async
  // executor (e.g. MooncakeMigrationExecutor) finishes its in-flight job during
  // destruction and fires its DoneCallback -> publishAck, which locks ackMutex
  // and uses ackProducer. Destroying it here, in the dtor body, guarantees
  // those members still exist. Relying on member-destruction order would be
  // fragile: `executor` is declared before `ackMutex`, so by default it is torn
  // down AFTER ackMutex -- the in-flight ack would then lock a destroyed mutex.
  executor.reset();
}

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

    const uint64_t kafkaRequestId = parsed->kafka_request_id;
    const std::optional<uint64_t> migrationId = parsed->migration_id;
    const tt::services::MigrationRequest apiReq{
        .src_slot = parsed->src_slot,
        .dst_slot = parsed->dst_slot,
        .layer_begin = parsed->layer_begin,
        .layer_end = parsed->layer_end,
        .src_position_begin = parsed->src_position_begin,
        .src_position_end = parsed->src_position_end,
        .dst_position_begin = parsed->dst_position_begin,
        .dst_position_end = parsed->dst_position_end,
        .migration_id = migrationId,
    };

    TT_LOG_DEBUG(
        "[KvMigrationWorker] dispatching kafka_request_id={} "
        "migration_id={} to executor",
        kafkaRequestId,
        migrationId.has_value() ? std::to_string(*migrationId) : "none");

    if (!executor) {
      // Surface the failure rather than silently dropping the request.
      publishAck(kafkaRequestId, migrationId,
                 tt::services::MigrationStatus::FAILED);
      continue;
    }

    // execute() is contractually non-blocking; the callback may fire on
    // this thread (synchronous Stub) or on an executor-owned thread.
    // Transport job uuid stays the per-request kafkaRequestId so parallel
    // layers of the same burst do not collide on the control channel.
    executor->execute(kafkaRequestId, apiReq,
                      [this, kafkaRequestId,
                       migrationId](tt::services::MigrationStatus status) {
                        publishAck(kafkaRequestId, migrationId, status);
                      });
  }

  TT_LOG_INFO("[KvMigrationWorker] consumer loop exited");
}

void KvMigrationWorker::publishAck(uint64_t kafkaRequestId,
                                   std::optional<uint64_t> migrationId,
                                   tt::services::MigrationStatus status) {
  const tt::messaging::MigrationResponseMessage ackMsg{
      .kafka_request_id = kafkaRequestId,
      .migration_id = migrationId,
      .status = status,
  };
  const std::string payload = tt::messaging::serialize(ackMsg);

  if (!ackProducer) {
    TT_LOG_ERROR(
        "[KvMigrationWorker] no ackProducer; cannot publish ack for "
        "kafka_request_id={} migration_id={}",
        kafkaRequestId,
        migrationId.has_value() ? std::to_string(*migrationId) : "none");
    return;
  }

  std::string err;
  bool sent = false;
  {
    // KafkaProducer::send is thread-safe at the librdkafka layer, but we
    // also want to serialize against any future producer-state mutation.
    std::lock_guard<std::mutex> lock(ackMutex);
    sent = ackPartition.has_value()
               ? ackProducer->send(payload, *ackPartition, &err)
               : ackProducer->send(payload, &err);
  }

  if (!sent) {
    TT_LOG_ERROR(
        "[KvMigrationWorker] ackProducer.send failed for kafka_request_id={} "
        "migration_id={}: {}",
        kafkaRequestId,
        migrationId.has_value() ? std::to_string(*migrationId) : "none", err);
  } else {
    TT_LOG_DEBUG(
        "[KvMigrationWorker] published ack kafka_request_id={} "
        "migration_id={} status={}",
        kafkaRequestId,
        migrationId.has_value() ? std::to_string(*migrationId) : "none",
        static_cast<int>(status));
  }
}

}  // namespace tt::worker
