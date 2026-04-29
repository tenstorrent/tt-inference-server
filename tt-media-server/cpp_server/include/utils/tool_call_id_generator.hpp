#pragma once

#include <random>
#include <string>

namespace tt::utils {

/**
 * Utility class for generating unique tool call IDs.
 *
 * Tool call IDs are strings in format "call_<24_random_chars>" where the
 * random chars are case-sensitive alphanumeric (a-z, A-Z, 0-9). This provides:
 * - Thread-safe generation via thread_local random engine
 * - Non-deterministic IDs for better uniqueness
 * - OpenAI-compatible format
 */
class ToolCallIDGenerator {
 public:
  /**
   * Generate a new unique tool call ID using 24 random alphanumeric characters.
   * Thread-safe, returns "call_<24_random_chars>".
   */
  static std::string generate() {
    static constexpr const char kChars[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    std::random_device rd;
    std::string result = "call_";
    result.reserve(29);

    for (int i = 0; i < 24; ++i) {
      result += kChars[rd() % 62];
    }

    return result;
  }
};

}  // namespace tt::utils
