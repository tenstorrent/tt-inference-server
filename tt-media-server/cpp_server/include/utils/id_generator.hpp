// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>

namespace tt::utils {

/**
 * Utility class for generating unique task IDs.
 *
 * Task IDs are uint32_t values generated via atomic counter.
 * This provides:
 * - Thread-safe generation
 * - Deterministic sequential IDs (1, 2, 3, ...)
 * - Simple debugging (sequential IDs are easy to track)
 */
class TaskIDGenerator {
 public:
  /**
   * Generate a new unique task ID using atomic counter.
   * Thread-safe, starts at 1 and increments.
   */
  static uint32_t generate() {
    static std::atomic<uint32_t> counter{0};
    return ++counter;
  }
};

}  // namespace tt::utils
