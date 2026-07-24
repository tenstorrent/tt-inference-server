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
 * one-sided over Mooncake. The data-plane sequence (RDMA-over-host bounce
 * buffer, windowed + credit-based):
 *
 *   TableExchange (once) -> BeginMigration -> BounceReady (segment + geometry)
 *   -> per window: [WRITEs into free sections] -> WindowReady (section
 * descriptors)
 *      -> WindowAck (receiver drained the window, returns credits)
 *   -> DoneMarker -> Ack.
 */
enum class KvControlType : uint8_t {
  TABLE_EXCHANGE = 1,
  BEGIN_MIGRATION = 2,
  DONE_MARKER = 3,
  ACK = 4,
  BOUNCE_READY = 5,  ///< receiver -> sender: bounce-buffer segment + geometry.
  WINDOW_READY = 6,  ///< sender -> receiver: sections written this window.
  WINDOW_ACK = 7,    ///< receiver -> sender: window drained, credits freed.
};

/// One device a bounce section's bytes drain to (a replica within a device
/// group). The size is per-descriptor (a merged section's targets share it), so
/// only the destination coordinates live here.
struct DrainTarget {
  uint32_t device = 0;    ///< LocalDeviceId (encodeDevice of the fabric node).
  uint64_t noc_addr = 0;  ///< channel << 32 | local_addr.

  bool operator==(const DrainTarget& o) const {
    return device == o.device && noc_addr == o.noc_addr;
  }
};

/// One filled bounce section and the device address(es) it drains to (fan-out).
struct BounceSectionDescriptor {
  uint64_t section_offset =
      0;              ///< Byte offset of the section within the bounce buffer.
  uint64_t size = 0;  ///< Contiguous bytes written into the section.
  std::vector<DrainTarget> targets;

  bool operator==(const BounceSectionDescriptor& o) const {
    return section_offset == o.section_offset && size == o.size &&
           targets == o.targets;
  }
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

  // BounceReady: the receiver's advertised Mooncake segment.
  std::string segment_name;

  // BounceReady/Ack/WindowAck: false signals the responder failed (bounce
  // buffer could not be registered, drain failed). Distinguishes failure from a
  // legitimately empty segment_name and lets the peer abort instead of
  // proceeding on bad state.
  bool ok = true;

  // TableExchange: 0 = prefill/sender table, 1 = decode/receiver table.
  uint8_t role = 0;
  std::vector<uint8_t> table_blob;  ///< Opaque serialized table.

  // BounceReady: geometry of the bounce buffer the sender fills.
  uint32_t bounce_section_count = 0;
  uint64_t bounce_section_size = 0;

  // WindowAck: number of sections the receiver drained and freed (credits).
  uint32_t credits = 0;

  // WindowReady: the sections the sender wrote into the bounce buffer this
  // window.
  std::vector<BounceSectionDescriptor> window;

  bool operator==(const KvControlMessage& o) const {
    return type == o.type && uuid == o.uuid && slot == o.slot &&
           layer_begin == o.layer_begin && layer_end == o.layer_end &&
           position_begin == o.position_begin &&
           position_end == o.position_end && segment_name == o.segment_name &&
           ok == o.ok && role == o.role && table_blob == o.table_blob &&
           bounce_section_count == o.bounce_section_count &&
           bounce_section_size == o.bounce_section_size &&
           credits == o.credits && window == o.window;
  }

  /// Serialize to a self-describing byte buffer (little-endian scalars).
  std::vector<uint8_t> serialize() const;

  /// Parse a buffer produced by serialize(); std::nullopt if malformed/short.
  static std::optional<KvControlMessage> deserialize(
      std::span<const uint8_t> bytes);
};

}  // namespace tt::transport
