// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tt::domain {

// TaskID is now a simple uint32_t for efficiency and simplicity
using TaskID = uint32_t;

// Utility class for TaskID generation and serialization
class TaskIDGenerator {
 public:
  static constexpr size_t K_SERIALIZED_SIZE = 4;

  // Generate a new unique TaskID using atomic counter
  static TaskID generate() {
    static std::atomic<uint32_t> counter{0};
    return ++counter;
  }

  // Serialize TaskID to 4-byte buffer for IPC
  static std::vector<char> serialize(TaskID taskId) {
    std::vector<char> buf(K_SERIALIZED_SIZE);
    std::memcpy(buf.data(), &taskId, K_SERIALIZED_SIZE);
    return buf;
  }

  // Deserialize TaskID from 4-byte buffer
  static TaskID deserialize(const char* data, size_t len) {
    if (len < K_SERIALIZED_SIZE) {
      return 0;  // Invalid/default TaskID
    }
    TaskID taskId;
    std::memcpy(&taskId, data, K_SERIALIZED_SIZE);
    return taskId;
  }
};

}  // namespace tt::domain
