// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "transport/kv_control_channel.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"

namespace tt::transport {

/**
 * @brief Drives the sender (prefill) side of a migration over the control
 *        channel.
 *
 * Sequences one migration end to end:
 *   BeginMigration -> (await) MirrorReady -> transferSlot (one-sided WRITEs)
 *   -> DoneMarker -> (await) Ack.
 *
 * Holds references to the control channel and the data-plane sender; owns no
 * threads — call migrate() from the prefill worker.
 */
class KvMigrationSender {
 public:
  KvMigrationSender(KvControlChannel& channel, MooncakeKvSender& sender);

  /**
   * @brief Init-time table exchange: send our serialized table, return the
   *        peer's. The caller deserializes the peer blob (e.g. via
   *        KvChunkAddressTableAdapter::fromProtobuf) into the decode table.
   * @return the peer's table blob, or std::nullopt on protocol error.
   */
  std::optional<std::vector<uint8_t>> exchangeTables(
      const std::vector<uint8_t>& localTableBlob);

  /// Run one migration to completion.
  ///
  /// @return true only if the whole slot landed on the decode device. false
  /// means the migration did NOT complete: the decode device may hold a partial
  /// mix of new and stale KV. The caller MUST NOT let the decode engine consume
  /// the slot until migrate() returns true, and recovers by retrying the *same*
  /// request (the drain is idempotent — see README "Contract for a higher-layer
  /// caller"). There is no rollback to prior contents.
  bool migrate(uint64_t uuid, const MigrationRequest& request);

 private:
  KvControlChannel& channel_;
  MooncakeKvSender& sender_;
};

/**
 * @brief Drives the receiver (decode) side of a migration over the control
 *        channel.
 *
 * Reacts to inbound control messages: BeginMigration -> prepareMirror +
 * MirrorReady; DoneMarker -> drain + Ack. run() services messages until the
 * channel closes; serveOne() handles a single message (for stepwise tests).
 */
class KvMigrationReceiver {
 public:
  KvMigrationReceiver(KvControlChannel& channel, MooncakeKvReceiver& receiver);

  /// Init-time table exchange: receive the peer's table, then send ours.
  std::optional<std::vector<uint8_t>> exchangeTables(
      const std::vector<uint8_t>& localTableBlob);

  /// Handle one inbound message. @return false when the channel has closed.
  bool serveOne();

  /// Service messages until the channel closes.
  void run();

 private:
  KvControlChannel& channel_;
  MooncakeKvReceiver& receiver_;
};

}  // namespace tt::transport
