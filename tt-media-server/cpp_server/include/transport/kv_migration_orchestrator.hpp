// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
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
 * Reacts to inbound control messages: TABLE_EXCHANGE (store peer prefill table
 * + reply with local decode table); BeginMigration -> prepareMirror +
 * MirrorReady; DoneMarker -> drain + Ack. run() services messages until the
 * channel closes, waiting indefinitely (idle) between requests; serveOne()
 * handles a single message with a bounded wait (for stepwise tests).
 */
class KvMigrationReceiver {
 public:
  /// @param localTableBlob this decode's serialized `.pb` (replied on
  ///        TABLE_EXCHANGE). Empty disables table provisioning on this channel.
  KvMigrationReceiver(KvControlChannel& channel, MooncakeKvReceiver& receiver,
                      std::vector<uint8_t> localTableBlob = {});

  /// Init-time table exchange: receive the peer's table, then send ours.
  /// Also stores the peer blob via peerTableBlob() / peerTable().
  std::optional<std::vector<uint8_t>> exchangeTables(
      const std::vector<uint8_t>& localTableBlob);

  /// Prefill `.pb` bytes from the last successful TABLE_EXCHANGE (empty until
  /// then).
  const std::vector<uint8_t>& peerTableBlob() const { return peer_table_blob_; }

  /// Parsed peer (prefill) table when the blob deserializes; nullptr otherwise.
  std::shared_ptr<const IKvTable> peerTable() const { return peer_table_; }

  /// Handle one inbound message (bounded wait). @return false when the channel
  /// has closed OR the wait times out — so a long idle gap ends serving. Kept
  /// for stepwise tests where the peer sends immediately; run() is the
  /// long-lived path that tolerates idle timeouts.
  bool serveOne();

  /// Service messages until the channel closes. Unlike serveOne(), an idle
  /// receive timeout is not a close: run() keeps waiting for the next request.
  void run();

 private:
  /// Dispatch a received message (TABLE_EXCHANGE, prepareMirror/MirrorReady,
  /// drain/Ack). @return false only if a reply send fails (peer would wedge —
  /// stop).
  bool handle(const KvControlMessage& msg);

  /// Keep the peer's TABLE_EXCHANGE blob (and parse when possible).
  void storePeerTable(std::vector<uint8_t> blob);

  /// Reply to an inbound TABLE_EXCHANGE after storing the peer blob.
  bool handleTableExchange(const KvControlMessage& msg);

  KvControlChannel& channel_;
  MooncakeKvReceiver& receiver_;
  std::vector<uint8_t> local_table_blob_;
  std::vector<uint8_t> peer_table_blob_;
  std::shared_ptr<const IKvTable> peer_table_;
};

}  // namespace tt::transport
