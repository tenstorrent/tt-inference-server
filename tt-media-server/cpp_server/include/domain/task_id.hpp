// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

namespace tt::domain {

struct TaskID {
  static constexpr size_t K_SERIALIZED_SIZE = sizeof(uint32_t);

  TaskID() = default;
  explicit TaskID(uint32_t taskId) : id(taskId) {}

  uint32_t id = 0;

  bool operator==(const TaskID& other) const { return id == other.id; }

  std::vector<char> ipcSerialize() const {
    std::vector<char> buf(K_SERIALIZED_SIZE);
    std::memcpy(buf.data(), &id, K_SERIALIZED_SIZE);
    return buf;
  }

  static TaskID ipcDeserialize(const char* data, size_t /*len*/) {
    uint32_t val = 0;
    std::memcpy(&val, data, sizeof(val));
    return TaskID(val);
  }

  static uint32_t generate() {
    static std::atomic<uint32_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
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
    return hash<uint32_t>{}(s.id);
  }
};
}  // namespace std
