// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cctype>
#include <cstdint>
#include <string>

namespace tt::utils {

struct ByteRange {
  enum class Kind {
    None,           // no usable Range header: serve the whole file (200)
    Partial,        // serve [offset, offset + length) (206)
    Unsatisfiable,  // 416
  };
  Kind kind = Kind::None;
  uint64_t offset = 0;
  uint64_t length = 0;
};

namespace detail {
inline bool allDigits(const std::string& s) {
  if (s.empty() || s.size() > 18) return false;
  for (unsigned char c : s) {
    if (!std::isdigit(c)) return false;
  }
  return true;
}
}  // namespace detail

/**
 * Single byte range of a `Range` header (RFC 7233) against a file of `size`
 * bytes: `bytes=a-b`, `bytes=a-`, `bytes=-n`. Anything else (another unit,
 * several ranges, bad syntax) is ignored, as the RFC allows, so the whole file
 * is served. A well formed range that starts past the end is unsatisfiable.
 */
inline ByteRange parseByteRange(const std::string& header, uint64_t size) {
  ByteRange none;
  std::string h;
  for (char c : header) {
    if (c != ' ' && c != '\t') h.push_back(c);
  }
  if (h.rfind("bytes=", 0) != 0) return none;
  const std::string spec = h.substr(6);
  if (spec.find(',') != std::string::npos) return none;
  const auto dash = spec.find('-');
  if (dash == std::string::npos) return none;
  const std::string first = spec.substr(0, dash);
  const std::string last = spec.substr(dash + 1);

  ByteRange out;
  if (first.empty()) {  // suffix: the last n bytes
    if (!detail::allDigits(last)) return none;
    const uint64_t n = std::stoull(last);
    if (n == 0 || size == 0) {
      out.kind = ByteRange::Kind::Unsatisfiable;
      return out;
    }
    out.kind = ByteRange::Kind::Partial;
    out.length = n >= size ? size : n;
    out.offset = size - out.length;
    return out;
  }
  if (!detail::allDigits(first)) return none;
  const uint64_t a = std::stoull(first);
  uint64_t b = size == 0 ? 0 : size - 1;
  if (!last.empty()) {
    if (!detail::allDigits(last)) return none;
    b = std::stoull(last);
    if (b < a) return none;  // invalid: ignore
  }
  if (a >= size) {
    out.kind = ByteRange::Kind::Unsatisfiable;
    return out;
  }
  if (b >= size) b = size - 1;
  out.kind = ByteRange::Kind::Partial;
  out.offset = a;
  out.length = b - a + 1;
  return out;
}

}  // namespace tt::utils
