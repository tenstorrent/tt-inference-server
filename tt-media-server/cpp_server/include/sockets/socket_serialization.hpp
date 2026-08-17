// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cstdint>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tt::sockets::wire {

template <typename T>
std::vector<uint8_t> serializeMessage(std::string_view messageType,
                                      const T& obj) {
  std::ostringstream oss;
  {
    cereal::BinaryOutputArchive archive(oss);
    std::string messageTypeString(messageType);
    archive(messageTypeString);
    obj.write(archive);
  }

  const std::string serialized = oss.str();
  return {serialized.begin(), serialized.end()};
}

inline std::string toSerializedString(std::span<const uint8_t> data) {
  if (data.empty()) {
    return {};
  }
  const auto* bytes = reinterpret_cast<const char*>(data.data());
  return {bytes, data.size()};
}

inline std::string readMessageType(std::span<const uint8_t> data) {
  std::string serialized = toSerializedString(data);
  std::istringstream iss(serialized);

  cereal::BinaryInputArchive archive(iss);
  std::string messageType;
  archive(messageType);
  return messageType;
}

template <typename T>
T deserializePayload(std::span<const uint8_t> data) {
  std::string serialized = toSerializedString(data);
  std::istringstream iss(serialized);

  cereal::BinaryInputArchive archive(iss);
  std::string ignoredMessageType;
  archive(ignoredMessageType);
  return T::read(archive);
}

}  // namespace tt::sockets::wire
