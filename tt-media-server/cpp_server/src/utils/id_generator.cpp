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
  // Additive golden-ratio (Weyl) sequence: each call advances a per-process
  // counter by 2^64/phi. Seeding with a random 64-bit value makes two restarts
  // of the same role produce different id streams. The high bits of a Weyl
  // sequence are low-discrepancy / equidistributed, so the truncated 8-hex ids
  // stay distinct within a process until ~2^32 requests (no random-birthday
  // clustering). Cross-process the 32-bit space still applies; pair with
  // task_id/timestamps for fleet-wide de-duplication.
  static const uint64_t seed = [] {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) | rd();
  }();
  static std::atomic<uint64_t> counter{seed};
  return counter.fetch_add(0x9E3779B97F4A7C15ULL,  // 2^64 / golden ratio
                           std::memory_order_relaxed);
}

}  // namespace

std::string TraceIdGenerator::generate() {
  return generate(tt::config::toString(tt::config::llmMode()));
}

std::string TraceIdGenerator::generate(std::string_view role) {
  uint64_t v = nextTraceCounter();
  // Render the high word as 8 hex chars: for a Weyl sequence the high bits
  // advance by the (near-irrational) golden-ratio stride, so consecutive ids
  // look unrelated rather than sequential.
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
