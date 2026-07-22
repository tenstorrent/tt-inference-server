// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/engine_table_handoff.hpp"

#include <utility>

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
  bool atEnd() const { return pos == bytes.size(); }

 private:
  std::span<const uint8_t> bytes;
  std::size_t pos = 0;
};

}  // namespace

std::vector<uint8_t> serializeEngineHandoff(const DeviceMap& deviceMap) {
  std::vector<uint8_t> out;
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
    TT_LOG_ERROR("[engine_table_handoff] trailing bytes after parse");
    return std::nullopt;
  }
  return payload;
}

bool sendEngineHandoff(sockets::ISocketTransport& transport,
                       const DeviceMap& deviceMap) {
  return transport.sendRawData(serializeEngineHandoff(deviceMap));
}

std::optional<DeviceMap> receiveEngineHandoff(
    sockets::ISocketTransport& transport) {
  const sockets::ReceiveResult result = transport.tryReceiveMessage();
  if (result.status == sockets::ReceiveStatus::NO_DATA) {
    return std::nullopt;
  }
  if (result.status == sockets::ReceiveStatus::CLOSED) {
    TT_LOG_ERROR("[engine_table_handoff] empty/closed receive");
    return std::nullopt;
  }
  auto payload = parseEngineHandoff(result.data);
  if (!payload) return std::nullopt;
  return std::move(payload->device_map);
}

SocketEngineDeviceMapSource::SocketEngineDeviceMapSource(
    std::shared_ptr<sockets::ISocketTransport> transport)
    : transport_(std::move(transport)) {}

std::optional<DeviceMap> SocketEngineDeviceMapSource::fetch() {
  if (!transport_) return std::nullopt;
  return receiveEngineHandoff(*transport_);
}

}  // namespace tt::transport
