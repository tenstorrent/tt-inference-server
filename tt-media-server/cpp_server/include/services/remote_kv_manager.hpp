// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <ctime>
#include <optional>

namespace tt::services {

enum class MigrationStatus {
  UNKNOWN,
  IN_PROGRESS,
  SUCCESSFUL,
  FAILED,
};

struct Migration {
  uint64_t migration_id;
  std::time_t time_created;
  MigrationStatus status;
};

/**
 * Range convention: all `_begin` / `_end` pairs are HALF-OPEN, i.e. [begin,
 * end).
 *
 * `migration_id` is the parent burst / sequence id (Sequence::getMigrationId).
 * When set, it is copied onto every Kafka request so workers and acks can be
 * grepped by the same end-to-end id. Per-request ack correlation still uses
 * the kafka_request_id returned by migrate().
 */
struct MigrationRequest {
  uint32_t src_slot;
  uint32_t dst_slot;
  uint32_t layer_begin;
  uint32_t layer_end;  // exclusive
  uint32_t src_position_begin;
  uint32_t src_position_end;  // exclusive
  uint32_t dst_position_begin;
  uint32_t dst_position_end;  // exclusive
  std::optional<uint64_t> migration_id;
};

/**
 * Async client to the pool of migration workers. The scheduler-facing
 * surface for issuing KV-cache migrations. Publishes requests on Kafka
 * and tracks completion via an ACK topic.
 */
class IRemoteKVManager {
 public:
  virtual ~IRemoteKVManager() = default;

  /**
   * Migrate KV Cache blocks. Returns immediately with a new unique id.
   * The actual transfer happens asynchronously on a remote worker.
   */
  [[nodiscard]] virtual uint64_t migrate(const MigrationRequest& request) = 0;

  /**
   * Look up the current status of a previously submitted migration.
   * Returns MigrationStatus::UNKNOWN if the id was never issued by
   * migrate() or has been garbage-collected.
   */
  virtual MigrationStatus getMigrationStatus(uint64_t migrationId) const = 0;
};

}  // namespace tt::services
