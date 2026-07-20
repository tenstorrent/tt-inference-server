// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_control_message.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace tt::transport {

namespace {

// Little-endian append helpers.
void putU8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }

void putU32(std::vector<uint8_t>& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void putU64(std::vector<uint8_t>& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void putBytes(std::vector<uint8_t>& out, const uint8_t* data, uint32_t len) {
  putU32(out, len);
  out.insert(out.end(), data, data + len);
}

// Bounds-checked little-endian reader over a byte span.
class Reader {
 public:
  explicit Reader(std::span<const uint8_t> bytes) : bytes(bytes) {}

  bool getU8(uint8_t& v) {
    if (pos + 1 > bytes.size()) return false;
    v = bytes[pos++];
    return true;
  }
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
  bool getBytes(std::vector<uint8_t>& out) {
    uint32_t len = 0;
    if (!getU32(len)) return false;
    if (pos + len > bytes.size()) return false;
    out.assign(bytes.begin() + pos, bytes.begin() + pos + len);
    pos += len;
    return true;
  }
  bool getString(std::string& out) {
    std::vector<uint8_t> tmp;
    if (!getBytes(tmp)) return false;
    out.assign(tmp.begin(), tmp.end());
    return true;
  }

  /// Bytes not yet consumed. Used to cap reservations against an
  /// attacker-controlled count before reading its elements.
  std::size_t remaining() const { return bytes.size() - pos; }

 private:
  std::span<const uint8_t> bytes;
  std::size_t pos = 0;
};

}  // namespace

std::vector<uint8_t> KvControlMessage::serialize() const {
  // Byte arrays are framed with a uint32 length prefix; refuse to emit a buffer
  // whose length would silently truncate on the cast below (which would desync
  // the receiver's parser). An empty return signals failure: a valid message
  // always has at least the type byte. Realistically unreachable, but
  // unbounded.
  constexpr std::size_t kMaxLen = std::numeric_limits<uint32_t>::max();
  if (segment_name.size() > kMaxLen || table_blob.size() > kMaxLen) {
    return {};
  }

  std::vector<uint8_t> out;
  putU8(out, static_cast<uint8_t>(type));
  putU64(out, uuid);
  putU32(out, slot);
  putU32(out, layer_begin);
  putU32(out, layer_end);
  putU32(out, position_begin);
  putU32(out, position_end);
  putBytes(out, reinterpret_cast<const uint8_t*>(segment_name.data()),
           static_cast<uint32_t>(segment_name.size()));
  putU8(out, ok ? 1 : 0);
  putU8(out, role);
  putBytes(out, table_blob.data(), static_cast<uint32_t>(table_blob.size()));
  // Bounce trailing fields (zero/empty on message types that don't use them).
  putU32(out, bounce_section_count);
  putU64(out, bounce_section_size);
  putU32(out, credits);
  putU32(out, static_cast<uint32_t>(window.size()));
  for (const BounceSectionDescriptor& d : window) {
    putU64(out, d.section_offset);
    putU64(out, d.size);
    putU32(out, static_cast<uint32_t>(d.targets.size()));
    for (const DrainTarget& t : d.targets) {
      putU32(out, t.device);
      putU64(out, t.noc_addr);
    }
  }
  return out;
}

std::optional<KvControlMessage> KvControlMessage::deserialize(
    std::span<const uint8_t> bytes) {
  Reader r(bytes);
  KvControlMessage m;
  uint8_t type = 0;
  if (!r.getU8(type)) return std::nullopt;
  if (type < static_cast<uint8_t>(KvControlType::TABLE_EXCHANGE) ||
      type > static_cast<uint8_t>(KvControlType::WINDOW_ACK)) {
    return std::nullopt;
  }
  m.type = static_cast<KvControlType>(type);
  if (!r.getU64(m.uuid)) return std::nullopt;
  if (!r.getU32(m.slot)) return std::nullopt;
  if (!r.getU32(m.layer_begin)) return std::nullopt;
  if (!r.getU32(m.layer_end)) return std::nullopt;
  if (!r.getU32(m.position_begin)) return std::nullopt;
  if (!r.getU32(m.position_end)) return std::nullopt;
  if (!r.getString(m.segment_name)) return std::nullopt;
  uint8_t ok = 0;
  if (!r.getU8(ok)) return std::nullopt;
  m.ok = ok != 0;
  if (!r.getU8(m.role)) return std::nullopt;
  if (!r.getBytes(m.table_blob)) return std::nullopt;
  if (!r.getU32(m.bounce_section_count)) return std::nullopt;
  if (!r.getU64(m.bounce_section_size)) return std::nullopt;
  if (!r.getU32(m.credits)) return std::nullopt;
  uint32_t windowLen = 0;
  if (!r.getU32(windowLen)) return std::nullopt;
  // Cap the reservation at what the remaining bytes could actually hold: a
  // descriptor is >= 20 wire bytes (u64 offset + u64 size + u32 target count),
  // so a count larger than remaining/20 is malformed. Without this, an
  // attacker-controlled uint32 count triggers a multi-GB reserve() (bad_alloc)
  // before any element is read. The loop below still enforces the true bound.
  constexpr std::size_t kMinDescriptorBytes = 20;
  m.window.reserve(
      std::min<std::size_t>(windowLen, r.remaining() / kMinDescriptorBytes));
  for (uint32_t i = 0; i < windowLen; ++i) {
    BounceSectionDescriptor d;
    if (!r.getU64(d.section_offset)) return std::nullopt;
    if (!r.getU64(d.size)) return std::nullopt;
    uint32_t targetLen = 0;
    if (!r.getU32(targetLen)) return std::nullopt;
    // Same cap for the per-descriptor target list: a DrainTarget is 12 wire
    // bytes (u32 device + u64 noc_addr).
    constexpr std::size_t kMinTargetBytes = 12;
    d.targets.reserve(
        std::min<std::size_t>(targetLen, r.remaining() / kMinTargetBytes));
    for (uint32_t j = 0; j < targetLen; ++j) {
      DrainTarget t;
      if (!r.getU32(t.device)) return std::nullopt;
      if (!r.getU64(t.noc_addr)) return std::nullopt;
      d.targets.push_back(t);
    }
    m.window.push_back(std::move(d));
  }
  return m;
}

}  // namespace tt::transport
