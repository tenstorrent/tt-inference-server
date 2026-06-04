// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/id_generator.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>

#include "config/settings.hpp"
#include "config/types.hpp"

namespace tt::utils {

namespace {

uint64_t nextTraceCounter() {
  // Per-process counter seeded with a random 64-bit value so two restarts of
  // the same role produce different id sequences. Combined with the role
  // prefix this is sufficient to keep "one greppable trace id per request"
  // (collision probability across two short-lived requests is ~2^-32 per
  // truncation to 8 hex chars and the role prefix narrows the search space
  // anyway).
  static const uint64_t seed = [] {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) | rd();
  }();
  static std::atomic<uint64_t> counter{seed};
  return counter.fetch_add(0x9E3779B97F4A7C15ULL,  // SplitMix64-style mix
                           std::memory_order_relaxed);
}

}  // namespace

std::string TraceIdGenerator::generate() {
  return generate(tt::config::toString(tt::config::llmMode()));
}

std::string TraceIdGenerator::generate(std::string_view role) {
  uint64_t v = nextTraceCounter();
  // Take 32 bits from the upper half (better distribution than the low bits
  // of a counter) and render as 8 hex chars.
  uint32_t hi = static_cast<uint32_t>(v >> 32);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%08x", hi);
  std::string out;
  out.reserve(role.size() + 1 + 8);
  out.append(role.data(), role.size());
  out.push_back('-');
  out.append(buf, 8);
  return out;
}

}  // namespace tt::utils
