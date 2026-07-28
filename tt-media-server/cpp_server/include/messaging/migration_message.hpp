// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "services/remote_kv_manager.hpp"

namespace tt::messaging {

/**
    Payload for a migration message. Consumed by migration worker.

    Range convention: all `_begin` / `_end` pairs are HALF-OPEN, i.e. [begin,
    end). `end` is exclusive.

    IDs:
    - `kafka_request_id` — unique per Kafka request/ack (per-layer fan-out).
    - `migration_id` — parent burst / sequence id (same name as on main /
      Sequence::getMigrationId). Optional on legacy payloads that only carried
      a per-request id under the old `migration_id` key.
*/
struct MigrationRequestMessage {
  uint64_t kafka_request_id;  // Per-request id, echoed back in the ack.
  std::optional<uint64_t> migration_id;  // Parent burst id when known.
  uint32_t src_slot;                     // Prefill (source) slot.
  uint32_t dst_slot;                     // Decode (destination) slot.
  uint32_t layer_begin;   // First layer, inclusive (shared by src and dst).
  uint32_t layer_end;     // One past the last layer, exclusive.
  uint32_t src_position_begin;  // Src token position start, inclusive.
  uint32_t src_position_end;    // Src token position end, exclusive.
  uint32_t dst_position_begin;  // Dst token position start, inclusive.
  uint32_t dst_position_end;    // Dst token position end, exclusive.
};

struct MigrationResponseMessage {
  uint64_t kafka_request_id;
  std::optional<uint64_t> migration_id;
  tt::services::MigrationStatus status;
};

std::string serialize(const MigrationRequestMessage& mrm);
std::string serialize(const MigrationResponseMessage& mrm);

std::optional<MigrationRequestMessage> parseMigrationRequest(
    const std::string& json);
std::optional<MigrationResponseMessage> parseMigrationResponse(
    const std::string& json);

}  // namespace tt::messaging
