// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <ctime>

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
 * What to migrate, prefill (src) -> decode (dst). Mirrors
 * tt::transport::MigrationRequest field-for-field so the scheduler-facing
 * surface, the Kafka wire message, and the data plane all speak one shape and
 * no request-shape translation is needed.
 *
 * The migration is asymmetric: a layer maps 1:1 (the layer range is shared),
 * but the slot and the token-position range may differ between sides (a
 * position shift / cross-slot move). The two position ranges must cover the
 * same number of chunks. Layer/position ranges are half-open [begin, end). A
 * symmetric whole-slot migration is src_slot==dst_slot and src positions==dst
 * positions.
 */
struct MigrationRequest {
  uint32_t src_slot;
  uint32_t dst_slot;
  uint32_t layer_begin;
  uint32_t layer_end;
  uint32_t src_position_begin;
  uint32_t src_position_end;
  uint32_t dst_position_begin;
  uint32_t dst_position_end;
};

/**
 * Async client to the pool of migration workers. The scheduler-facing
 * surface for issuing KV-cache migrations.Publishes requests on Kafka
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
  virtual MigrationStatus getStatus(uint64_t migrationId) const = 0;
};

}  // namespace tt::services