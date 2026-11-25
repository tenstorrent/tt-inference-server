// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <random>
#include <string>

namespace tt::utils {

/** Random (version 4) UUID, lower case, 8-4-4-4-12. Thread-safe. */
inline std::string generateUuidV4() {
  static std::mutex mutex;
  static std::mt19937_64 rng{[] {
    std::random_device device;
    std::seed_seq seed{device(), device(), device(), device()};
    return std::mt19937_64(seed);
  }()};
  uint64_t hi = 0;
  uint64_t lo = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    hi = rng();
    lo = rng();
  }
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;  // version 4
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;  // variant 10
  char buf[37];
  std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
                static_cast<unsigned>(hi >> 32),
                static_cast<unsigned>((hi >> 16) & 0xFFFF),
                static_cast<unsigned>(hi & 0xFFFF),
                static_cast<unsigned>(lo >> 48),
                static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return buf;
}

/** True for the 8-4-4-4-12 hex form (any case). */
inline bool isUuid(const std::string& text) {
  if (text.size() != 36) return false;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (c != '-') return false;
    } else if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                 (c >= 'A' && c <= 'F'))) {
      return false;
    }
  }
  return true;
}

}  // namespace tt::utils
