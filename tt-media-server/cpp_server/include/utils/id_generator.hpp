// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <random>

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

/**
 * Generates unique 64-bit migration IDs using a thread-local PRNG.
 * Used to correlate prefill requests with their results across the
 * inter-server socket.
 */
class MigrationIDGenerator {
 public:
  static uint64_t generate() {
    thread_local std::mt19937_64 gen = []() {
      std::random_device rd;
      return std::mt19937_64(rd());
    }();
    thread_local std::uniform_int_distribution<uint64_t> dist;
    return dist(gen);
  }
};

}  // namespace tt::utils
