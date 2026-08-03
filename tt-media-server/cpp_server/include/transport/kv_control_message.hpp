// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace tt::transport {

/**
 * @brief Control-plane message kinds for KV migration.
 *
 * The control channel carries setup and coordination; the bulk KV bytes move
 * one-sided over Mooncake. Sequence (per migration):
 *   TableExchange (once, at connection setup, both directions)
 *   -> BeginMigration (sender -> receiver: slot + layer/position ranges)
 *   -> MirrorReady    (receiver -> sender: the segment to write into)
 *   -> [Mooncake one-sided WRITEs of each chunk]
 *   -> DoneMarker     (sender -> receiver: all chunks written)
 *   -> Ack            (receiver -> sender: drained mirror -> device)
 */
enum class KvControlType : uint8_t {
  TABLE_EXCHANGE = 1,
  BEGIN_MIGRATION = 2,
  MIRROR_READY = 3,
  DONE_MARKER = 4,
  ACK = 5,
};

/**
 * @brief A control-plane message. A single tagged struct (rather than a class
 *        hierarchy) keeps the wire codec simple; only the fields relevant to
 *        `type` are meaningful, but all are serialized for a fixed layout.
 */
struct KvControlMessage {
  KvControlType type = KvControlType::ACK;
  uint64_t uuid = 0;

  // BeginMigration: what to migrate.
  uint32_t slot = 0;
  uint32_t layer_begin = 0;
  uint32_t layer_end = 0;
  uint32_t position_begin = 0;
  uint32_t position_end = 0;

  // MirrorReady: the receiver's advertised Mooncake segment.
  std::string segment_name;

  // MirrorReady/Ack: false signals the responder failed (mirror could not be
  // prepared, drain failed). Distinguishes failure from a legitimately empty
  // segment_name and lets the peer abort instead of proceeding on bad state.
  bool ok = true;

  // TableExchange: 0 = prefill/sender table, 1 = decode/receiver table.
  uint8_t role = 0;
  std::vector<uint8_t> table_blob;  ///< Opaque serialized table.

  bool operator==(const KvControlMessage& o) const {
    return type == o.type && uuid == o.uuid && slot == o.slot &&
           layer_begin == o.layer_begin && layer_end == o.layer_end &&
           position_begin == o.position_begin &&
           position_end == o.position_end && segment_name == o.segment_name &&
           ok == o.ok && role == o.role && table_blob == o.table_blob;
  }

  /// Serialize to a self-describing byte buffer (little-endian scalars).
  std::vector<uint8_t> serialize() const;

  /// Parse a buffer produced by serialize(); std::nullopt if malformed/short.
  static std::optional<KvControlMessage> deserialize(
      std::span<const uint8_t> bytes);
};

}  // namespace tt::transport
