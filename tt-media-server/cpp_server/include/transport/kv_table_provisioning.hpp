// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "transport/kv_control_channel.hpp"
#include "transport/kv_table_view.hpp"

namespace tt::transport {

/**
 * @brief Real KV-table loading + init-time exchange over the control channel.
 *
 * Each worker loads ONLY its own `.pb` from disk, then both sides swap
 * serialized tables over the control channel (TABLE_EXCHANGE). Prefill keeps
 * the peer decode table for destination addressing; decode keeps the peer
 * prefill table from the same exchange.
 *
 * The exchanged blob is just the `.pb` file's bytes; the peer reconstructs the
 * table with `KvChunkAddressTableAdapter::fromProtobuf` (which is a no-op
 * returning nullptr when ENABLE_KV_TABLE is OFF, so this all still links in
 * every build). These are free functions — usable before the sender object
 * exists, which it must be: the sender can't be constructed until the decode
 * table arrives.
 */

/// Which end of the control channel we are. The two roles use opposite
/// send/receive ordering so a single channel exchange never deadlocks.
enum class TableExchangeRole {
  Sender,    ///< prefill: send local table, then receive the peer's.
  Receiver,  ///< decode: receive the peer's table, then send local.
};

struct LoadedKvTable {
  std::shared_ptr<const IKvTable>
      table;                  ///< parsed table (nullptr on failure).
  std::vector<uint8_t> blob;  ///< the `.pb` bytes, to exchange.
};

/// Read a serialized KvChunkAddressTable `.pb` and parse it. Returns nullopt if
/// the file is unreadable/empty or parsing fails (e.g. ENABLE_KV_TABLE is OFF).
std::optional<LoadedKvTable> loadKvTableFile(const std::string& path);

/// Parse a serialized-table blob into an IKvTable. nullptr on empty/invalid
/// input or when the table guard is off.
std::shared_ptr<const IKvTable> deserializeKvTable(
    const std::vector<uint8_t>& blob);

/// Swap serialized-table blobs over `channel` (one TABLE_EXCHANGE each way).
/// @return the peer's blob, or nullopt on protocol error / channel close.
std::optional<std::vector<uint8_t>> exchangeTableBlob(
    KvControlChannel& channel, TableExchangeRole role,
    const std::vector<uint8_t>& localBlob);

/// exchangeTableBlob + deserialize: the peer's table, or nullptr on failure.
/// The prefill side calls this (role=Sender) to obtain the decode table before
/// building its MooncakeKvSender / KvMigrationMultiHostSender.
std::shared_ptr<const IKvTable> provisionPeerTable(
    KvControlChannel& channel, TableExchangeRole role,
    const std::vector<uint8_t>& localBlob);

}  // namespace tt::transport
