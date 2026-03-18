// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <algorithm>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace tt::domain {

struct TaskID {
  static constexpr size_t K_SERIALIZED_SIZE = 36;

  TaskID() = default;
  explicit TaskID(std::string taskId) : id(std::move(taskId)) {}

  std::string id;

  bool operator==(const TaskID& other) const { return id == other.id; }

  std::vector<char> ipcSerialize() const {
    std::vector<char> buf(K_SERIALIZED_SIZE, '\0');
    std::copy_n(id.begin(), std::min(id.size(), K_SERIALIZED_SIZE),
                buf.begin());
    return buf;
  }

  static TaskID ipcDeserialize(const char* data, size_t len) {
    size_t actualLen = strnlen(data, len);
    return TaskID(std::string(data, actualLen));
  }

  static std::string generate() {
    return boost::uuids::to_string(boost::uuids::random_generator()());
  }
};

inline std::ostream& operator<<(std::ostream& os, const TaskID& tid) {
  return os << tid.id;
}

}  // namespace tt::domain

namespace std {
template <>
struct hash<tt::domain::TaskID> {
  size_t operator()(const tt::domain::TaskID& s) const {
    return hash<string>{}(s.id);
  }
};
}  // namespace std
