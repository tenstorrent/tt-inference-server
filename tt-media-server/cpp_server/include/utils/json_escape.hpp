// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <cstdio>
#include <string>

namespace tt::utils {

/**
 * Escape a string for safe embedding inside a JSON string literal.
 * Handles: \, ", and control characters U+0000–U+001F.
 */
inline std::string jsonEscape(const std::string& s) {
  std::string result;
  result.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        result.append("\\\"");
        break;
      case '\\':
        result.append("\\\\");
        break;
      case '\b':
        result.append("\\b");
        break;
      case '\f':
        result.append("\\f");
        break;
      case '\n':
        result.append("\\n");
        break;
      case '\r':
        result.append("\\r");
        break;
      case '\t':
        result.append("\\t");
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          result.append(buf);
        } else {
          result.push_back(static_cast<char>(c));
        }
    }
  }
  return result;
}

}  // namespace tt::utils
