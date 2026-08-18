// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "transport/kv_control_channel.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"

namespace tt::transport {

/**
 * @brief Drives the sender (prefill) side of a bounce migration over the
 * control channel.
 *
 * Sequences one migration:
 *   BeginMigration -> (await) BounceReady -> per window: WindowReady ->
 *   (await) WindowAck -> ... -> DoneMarker -> (await) Ack.
 *
 * The window handshake is the WindowSink handed to the data-plane sender, so
 * the sender stays channel-free and the credit-based backpressure lives here.
 */
class KvMigrationSender {
 public:
  KvMigrationSender(KvControlChannel& channel, MooncakeKvSender& sender);

  /// Init-time table exchange (send ours, return the peer's).
  std::optional<std::vector<uint8_t>> exchangeTables(
      const std::vector<uint8_t>& localTableBlob);

  /// Run one migration to completion. @return true only if the whole slot
  /// landed on the decode device (same contract as KvMigrationSender::migrate).
  bool migrate(uint64_t uuid, const MigrationRequest& request);

 private:
  KvControlChannel& channel_;
  MooncakeKvSender& sender_;
};

/**
 * @brief Drives the receiver (decode) side of a bounce migration.
 *
 * Reacts to inbound control messages: TABLE_EXCHANGE (reply local decode
 * table); BeginMigration -> BounceReady (segment + geometry); WindowReady ->
 * drainWindow + WindowAck; DoneMarker -> Ack (carrying whether every window
 * drained). Holds no table of its own — the bounce receiver drains purely from
 * window descriptors.
 */
class KvMigrationReceiver {
 public:
  KvMigrationReceiver(
      KvControlChannel& channel, MooncakeKvReceiver& receiver,
      std::shared_ptr<const std::vector<uint8_t>> localTableBlob = nullptr);
  /// A null receiver enables control-only dry-run mode: TABLE_EXCHANGE is
  /// served, while migration requests are logged and rejected.
  KvMigrationReceiver(
      KvControlChannel& channel, MooncakeKvReceiver* receiver,
      std::shared_ptr<const std::vector<uint8_t>> localTableBlob = nullptr);

  /// Init-time table exchange: receive the peer's table, then send ours.
  std::optional<std::vector<uint8_t>> exchangeTables(
      const std::vector<uint8_t>& localTableBlob);

  /// Handle one inbound message (bounded wait). @return false on close/timeout.
  bool serveOne();

  /// Service messages until the channel closes (idle timeouts keep waiting).
  void run();

 private:
  bool handle(const KvControlMessage& msg);
  bool handleTableExchange(const KvControlMessage& msg);

  KvControlChannel& channel_;
  MooncakeKvReceiver* receiver_;
  std::shared_ptr<const std::vector<uint8_t>> local_table_blob_;
  // uuid -> whether every window so far drained ok (reset on BeginMigration).
  std::unordered_map<uint64_t, bool> ok_;
};

}  // namespace tt::transport
