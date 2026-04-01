// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tt::utils {

/**
 * Utility class for generating unique task IDs and serialization.
 *
 * Task IDs are uint32_t values generated via atomic counter.
 * This provides:
 * - Thread-safe generation
 * - Deterministic sequential IDs (1, 2, 3, ...)
 * - Compact 4-byte serialization for IPC
 * - Simple debugging (sequential IDs are easy to track)
 */
class TaskIDGenerator {
 public:
  static constexpr size_t K_SERIALIZED_SIZE = 4;

  /**
   * Generate a new unique task ID using atomic counter.
   * Thread-safe, starts at 1 and increments.
   */
  static uint32_t generate() {
    static std::atomic<uint32_t> counter{0};
    return ++counter;
  }

  /**
   * Serialize task ID to 4-byte buffer for IPC communication.
   * Uses simple memcpy (assumes local IPC, no endianness concerns).
   */
  static std::vector<char> serialize(uint32_t taskId) {
    std::vector<char> buf(K_SERIALIZED_SIZE);
    std::memcpy(buf.data(), &taskId, K_SERIALIZED_SIZE);
    return buf;
  }

  /**
   * Deserialize task ID from 4-byte buffer.
   * Returns 0 if buffer is too small (invalid ID).
   */
  static uint32_t deserialize(const char* data, size_t len) {
    if (len < K_SERIALIZED_SIZE) {
      return 0;  // Invalid/default task ID
    }
    uint32_t taskId;
    std::memcpy(&taskId, data, K_SERIALIZED_SIZE);
    return taskId;
  }
};

}  // namespace tt::utils
