// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_table_provisioning.hpp"

#include <fstream>
#include <utility>

#include "transport/kv_chunk_address_table_adapter.hpp"
#include "transport/kv_control_message.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

std::shared_ptr<const IKvTable> deserializeKvTable(
    const std::vector<uint8_t>& blob) {
  if (blob.empty()) {
    return nullptr;
  }
  // fromProtobuf returns the no-op-fallback nullptr when ENABLE_KV_TABLE is OFF.
  std::unique_ptr<KvChunkAddressTableAdapter> adapter =
      KvChunkAddressTableAdapter::fromProtobuf(
          std::string(blob.begin(), blob.end()));
  return std::shared_ptr<const IKvTable>(std::move(adapter));
}

std::optional<LoadedKvTable> loadKvTableFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    TT_LOG_ERROR("[kv_table_provisioning] cannot open table file: {}", path);
    return std::nullopt;
  }
  const std::streamoff size = file.tellg();
  if (size <= 0) {
    TT_LOG_ERROR("[kv_table_provisioning] empty/unsized table file: {}", path);
    return std::nullopt;
  }

  std::vector<uint8_t> blob(static_cast<std::size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(blob.data()), size);
  if (!file) {
    TT_LOG_ERROR("[kv_table_provisioning] short read on table file: {}", path);
    return std::nullopt;
  }

  auto table = deserializeKvTable(blob);
  if (!table) {
    TT_LOG_ERROR(
        "[kv_table_provisioning] failed to parse '{}' (bad .pb, or "
        "ENABLE_KV_TABLE is OFF)",
        path);
    return std::nullopt;
  }
  TT_LOG_INFO("[kv_table_provisioning] loaded table '{}' ({} bytes)", path,
              blob.size());
  return LoadedKvTable{std::move(table), std::move(blob)};
}

std::optional<std::vector<uint8_t>> exchangeTableBlob(
    KvControlChannel& channel, TableExchangeRole role,
    const std::vector<uint8_t>& localBlob) {
  auto sendLocal = [&]() -> bool {
    KvControlMessage out;
    out.type = KvControlType::TABLE_EXCHANGE;
    out.role = (role == TableExchangeRole::Sender) ? 0 : 1;
    out.table_blob = localBlob;
    return channel.send(out);
  };
  auto receivePeer = [&]() -> std::optional<std::vector<uint8_t>> {
    const auto peer = channel.receive();
    if (!peer || peer->type != KvControlType::TABLE_EXCHANGE) {
      TT_LOG_ERROR(
          "[kv_table_provisioning] exchange: bad/absent peer table message");
      return std::nullopt;
    }
    return peer->table_blob;
  };

  // Opposite ordering per role so a single bidirectional channel never
  // deadlocks: the sender writes first, the receiver reads first.
  if (role == TableExchangeRole::Sender) {
    if (!sendLocal()) {
      return std::nullopt;
    }
    return receivePeer();
  }
  auto peer = receivePeer();
  if (!peer) {
    return std::nullopt;
  }
  if (!sendLocal()) {
    return std::nullopt;
  }
  return peer;
}

std::shared_ptr<const IKvTable> provisionPeerTable(
    KvControlChannel& channel, TableExchangeRole role,
    const std::vector<uint8_t>& localBlob) {
  auto peerBlob = exchangeTableBlob(channel, role, localBlob);
  if (!peerBlob) {
    return nullptr;
  }
  return deserializeKvTable(*peerBlob);
}

}  // namespace tt::transport
