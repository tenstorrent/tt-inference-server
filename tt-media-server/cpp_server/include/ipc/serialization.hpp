// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace tt::ipc::serialization {

inline void writeString(std::ostream& os, const std::string& value) {
  const uint32_t size = static_cast<uint32_t>(value.size());
  os.write(reinterpret_cast<const char*>(&size), sizeof(size));
  os.write(value.data(), static_cast<std::streamsize>(value.size()));
}

inline std::string readString(std::istream& is) {
  uint32_t size = 0;
  is.read(reinterpret_cast<char*>(&size), sizeof(size));
  std::string value(size, '\0');
  if (size > 0) {
    is.read(value.data(), static_cast<std::streamsize>(size));
  }
  return value;
}

template <typename T>
inline void writeVector(std::ostream& os, const std::vector<T>& values) {
  const uint32_t size = static_cast<uint32_t>(values.size());
  os.write(reinterpret_cast<const char*>(&size), sizeof(size));
  if (size > 0) {
    os.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(T)));
  }
}

template <typename T>
inline std::vector<T> readVector(std::istream& is) {
  uint32_t size = 0;
  is.read(reinterpret_cast<char*>(&size), sizeof(size));
  std::vector<T> values(size);
  if (size > 0) {
    is.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  }
  return values;
}

}  // namespace tt::ipc::serialization
