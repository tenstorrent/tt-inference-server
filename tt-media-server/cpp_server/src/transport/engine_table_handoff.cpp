// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/engine_table_handoff.hpp"

#include <utility>

#include "transport/kv_table_provisioning.hpp"  // deserializeKvTable
#include "utils/logger.hpp"

namespace tt::transport {

namespace {

void putU32(std::vector<uint8_t>& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void putU64(std::vector<uint8_t>& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

class Reader {
 public:
  explicit Reader(std::span<const uint8_t> input) : bytes(input) {}
  bool getU32(uint32_t& v) {
    if (pos + 4 > bytes.size()) return false;
    v = 0;
    for (int i = 0; i < 4; ++i)
      v |= static_cast<uint32_t>(bytes[pos++]) << (8 * i);
    return true;
  }
  bool getU64(uint64_t& v) {
    if (pos + 8 > bytes.size()) return false;
    v = 0;
    for (int i = 0; i < 8; ++i)
      v |= static_cast<uint64_t>(bytes[pos++]) << (8 * i);
    return true;
  }
  bool getBytes(std::vector<uint8_t>& out, uint32_t len) {
    if (pos + len > bytes.size()) return false;
    out.assign(bytes.begin() + pos, bytes.begin() + pos + len);
    pos += len;
    return true;
  }
  bool atEnd() const { return pos == bytes.size(); }

 private:
  std::span<const uint8_t> bytes;
  std::size_t pos = 0;
};

}  // namespace

std::vector<uint8_t> serializeEngineHandoff(
    const std::vector<uint8_t>& tableBlob, const DeviceMap& deviceMap) {
  std::vector<uint8_t> out;
  putU32(out, static_cast<uint32_t>(tableBlob.size()));
  out.insert(out.end(), tableBlob.begin(), tableBlob.end());

  putU32(out, static_cast<uint32_t>(deviceMap.size()));
  for (const auto& [device, umdChipId] : deviceMap.entries()) {
    // Invert encodeDevice = (mesh << 16) | chip so the engine's (mesh, chip)
    // coordinates round-trip.
    putU32(out, device >> 16);     // mesh
    putU32(out, device & 0xFFFF);  // chip
    putU64(out, umdChipId);
  }
  return out;
}

std::optional<EngineHandoffPayload> parseEngineHandoff(
    std::span<const uint8_t> bytes) {
  Reader r(bytes);
  EngineHandoffPayload payload;

  uint32_t tableLen = 0;
  if (!r.getU32(tableLen)) return std::nullopt;
  if (!r.getBytes(payload.table_blob, tableLen)) return std::nullopt;

  uint32_t count = 0;
  if (!r.getU32(count)) return std::nullopt;
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t mesh = 0;
    uint32_t chip = 0;
    uint64_t umdChipId = 0;
    if (!r.getU32(mesh) || !r.getU32(chip) || !r.getU64(umdChipId)) {
      return std::nullopt;
    }
    payload.device_map.set(FabricNode{mesh, chip}, umdChipId);
  }
  if (!r.atEnd()) {
    // Trailing garbage means a framing/version mismatch — reject rather than
    // silently accept a partially-understood message.
    TT_LOG_ERROR("[engine_table_handoff] trailing bytes after parse");
    return std::nullopt;
  }
  return payload;
}

bool sendEngineHandoff(sockets::ISocketTransport& transport,
                       const std::vector<uint8_t>& tableBlob,
                       const DeviceMap& deviceMap) {
  const auto bytes = serializeEngineHandoff(tableBlob, deviceMap);
  return transport.sendRawData(bytes);
}

std::optional<EngineTables> engineTablesFromWire(
    std::span<const uint8_t> bytes) {
  auto payload = parseEngineHandoff(bytes);
  if (!payload) {
    return std::nullopt;
  }
  // Keep wire bytes before deserialize so TABLE_EXCHANGE still has a .pb blob.
  std::vector<uint8_t> tableBlob = std::move(payload->table_blob);
  auto table = deserializeKvTable(tableBlob);
  if (!table) {
    TT_LOG_ERROR(
        "[engine_table_handoff] table failed to parse (bad bytes, or "
        "ENABLE_KV_TABLE is OFF)");
    return std::nullopt;
  }
  return EngineTables{std::move(table), std::move(tableBlob),
                      std::move(payload->device_map)};
}

std::optional<EngineTables> receiveEngineHandoff(
    sockets::ISocketTransport& transport) {
  const sockets::ReceiveResult result = transport.tryReceiveMessage();
  if (result.status == sockets::ReceiveStatus::NO_DATA) {
    return std::nullopt;
  }
  if (result.status == sockets::ReceiveStatus::CLOSED) {
    TT_LOG_ERROR("[engine_table_handoff] empty/closed receive");
    return std::nullopt;
  }
  return engineTablesFromWire(result.data);
}

SocketEngineTableSource::SocketEngineTableSource(
    std::shared_ptr<sockets::ISocketTransport> transport)
    : transport_(std::move(transport)) {}

std::optional<EngineTables> SocketEngineTableSource::fetch() {
  if (!transport_) return std::nullopt;
  return receiveEngineHandoff(*transport_);
}

}  // namespace tt::transport
