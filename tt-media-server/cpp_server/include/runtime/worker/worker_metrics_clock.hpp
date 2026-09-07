// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <chrono>
#include <cstdint>

namespace tt::worker {

/**
 * Clock for the heartbeat stamps workers write to shared memory and renderers
 * read back. Shared so writer and readers cannot disagree on the epoch.
 */
inline uint64_t nowMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

/**
 * Age of a stamp, clamped to 0 for the two cases a scrape legitimately sees: a
 * cell never written (0) and a stamp ahead of the read (writer raced reader).
 */
inline double ageSeconds(uint64_t lastEpochMs, uint64_t nowEpochMs) {
  if (lastEpochMs == 0 || lastEpochMs > nowEpochMs) return 0.0;
  return static_cast<double>(nowEpochMs - lastEpochMs) / 1000.0;
}

}  // namespace tt::worker
