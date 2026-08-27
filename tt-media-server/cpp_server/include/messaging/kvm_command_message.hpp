// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "services/remote_kv_manager.hpp"

namespace tt::messaging {

/**
 * Wire messages for the ZMQ-backed KV Manager control plane.
 *
 * These mirror the flat JSON schema that kv_manager already parses in
 * `kvm::cp::command::parseMigrationRequest` /
 * `kvm::cp::command::parseMigrationResponse` (tt-d-gen/kv_manager). The
 * schema is intentionally distinct from the Kafka wire (`kafka_request_id`)
 * so the two transports can evolve independently: kv_manager owns the ZMQ
 * wire and uses `command_id` for the per-request correlation id and
 * `migration_id` for the burst parent id.
 *
 * Range convention: all `_begin` / `_end` pairs are HALF-OPEN, i.e.
 * [begin, end); `end` is exclusive.
 */
struct KvmCommandMessage {
  uint64_t command_id;
  uint64_t migration_id;
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
 * Ack echoed by kv_manager on the reply topic. The status enum is shared
 * with the Kafka worker (`tt::services::MigrationStatus`) so
 * `RemoteKVManagerZmqImpl` can reuse the same status bookkeeping without a
 * translation layer.
 *
 * `command_id` is optional on the kv_manager wire (defaults to 0 when
 * absent), but our implementation always sets it — the field carries the
 * per-request correlation id that we generate on `migrate()`.
 */
struct KvmResponseMessage {
  uint64_t command_id;
  uint64_t migration_id;
  tt::services::MigrationStatus status;
};

/**
 * Serialize a command for the wire. The output is the flat JSON payload
 * body only — the topic prefix (see `IKvmZmqTransport`) is sent as a
 * separate ZMQ frame ahead of it and is not encoded here.
 */
std::string serialize(const KvmCommandMessage& msg);
std::string serialize(const KvmResponseMessage& msg);

std::optional<KvmCommandMessage> parseKvmCommand(const std::string& json);
std::optional<KvmResponseMessage> parseKvmResponse(const std::string& json);

}  // namespace tt::messaging
