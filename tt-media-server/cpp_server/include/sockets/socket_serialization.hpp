// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace tt::sockets::wire {

template <typename T>
std::vector<uint8_t> serializeMessage(const std::string& messageType,
                                      const T& obj) {
  std::ostringstream oss;
  {
    cereal::BinaryOutputArchive archive(oss);
    archive(messageType);
    obj.write(archive);
  }

  const std::string serialized = oss.str();
  return {serialized.begin(), serialized.end()};
}

inline std::string readMessageType(const std::vector<uint8_t>& data) {
  std::string serialized(data.begin(), data.end());
  std::istringstream iss(serialized);

  cereal::BinaryInputArchive archive(iss);
  std::string messageType;
  archive(messageType);
  return messageType;
}

template <typename T>
T deserializePayload(const std::vector<uint8_t>& data) {
  std::string serialized(data.begin(), data.end());
  std::istringstream iss(serialized);

  cereal::BinaryInputArchive archive(iss);
  std::string ignoredMessageType;
  archive(ignoredMessageType);
  return T::read(archive);
}

}  // namespace tt::sockets::wire
