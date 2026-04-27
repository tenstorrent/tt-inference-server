#pragma once

#include <atomic>
#include <string>

namespace tt::utils {

/**
 * Utility class for generating unique tool call IDs.
 *
 * Tool call IDs are strings in format "call_<N>" where N is a sequential number.
 * This provides:
 * - Thread-safe generation via atomic counter
 * - Deterministic sequential IDs (call_1, call_2, call_3, ...)
 * - Simple debugging (sequential IDs are easy to track in logs)
 * - OpenAI-compatible format
 */
class ToolCallIDGenerator {
 public:
  /**
   * Generate a new unique tool call ID using atomic counter.
   * Thread-safe, returns "call_1", "call_2", "call_3", etc.
   */
  static std::string generate() {
    static std::atomic<uint64_t> counter{0};
    return "call_" + std::to_string(++counter);
  }
};

}  // namespace tt::utils
