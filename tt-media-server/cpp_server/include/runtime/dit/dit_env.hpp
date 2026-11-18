// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace tt::dit {

inline std::string lowered(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

inline std::string trimmed(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

/** setenv() only when the value differs. The environment is process-wide and
 *  getenv() is not safe against a concurrent setenv() that adds a variable, so
 *  the runner thread must not modify what the server already prepared. */
inline void setEnvIfChanged(const char* name, const std::string& value) {
  const char* current = std::getenv(name);
  if (current != nullptr && value == current) return;
  ::setenv(name, value.c_str(), 1);
}

inline std::optional<std::string> envString(const char* name) {
  const char* v = std::getenv(name);
  if (!v) return std::nullopt;
  return std::string(v);
}

/** Unset or empty: `fallback`; "0", "false", "no", "off": false; else true. */
inline bool envFlag(const char* name, bool fallback) {
  const auto v = envString(name);
  if (!v || trimmed(*v).empty()) return fallback;
  const std::string k = lowered(trimmed(*v));
  return !(k == "0" || k == "false" || k == "no" || k == "off");
}

/** "1"/"true" -> true, "0"/"false" -> false, unset -> nullopt, else throws. */
inline std::optional<bool> envBoolOrNull(const char* name) {
  const auto v = envString(name);
  if (!v) return std::nullopt;
  const std::string k = lowered(trimmed(*v));
  if (k == "1" || k == "true") return true;
  if (k == "0" || k == "false") return false;
  throw std::invalid_argument(std::string(name) + "='" + *v +
                              "' must be 1/true or 0/false");
}

}  // namespace tt::dit
