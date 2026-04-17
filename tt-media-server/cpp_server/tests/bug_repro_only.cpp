// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Standalone reproducer for the TOCTOU bug in evictOldSessions (pre-fix).
// Uses ONLY ConcurrentMap::take (no takeIf dependency) — can compile against
// both the buggy and fixed codebase.
//
// Demonstrates that the original eviction logic (forEach then take, with no
// re-check of the in-flight flag) always evicts an in-flight session when the
// race window is forced open with a barrier.

#include <gtest/gtest.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "utils/concurrent_map.hpp"

namespace {

struct Session {
  bool in_flight{false};
  int slot_id{-1};
};

// Two-phase barrier forces the TOCTOU window open reliably.
struct Barrier {
  std::mutex mu;
  std::condition_variable cv;
  int phase{0};

  void waitFor(int p) {
    std::unique_lock lk(mu);
    cv.wait(lk, [&] { return phase >= p; });
  }
  void advance() {
    std::lock_guard lk(mu);
    ++phase;
    cv.notify_all();
  }
};

// Mirrors the ORIGINAL evictOldSessions logic (using take, not takeIf).
// The barrier makes the race window deterministic:
//   1. Scan (forEach) completes and releases the sessions lock.
//   2. Barrier phase 0→1: evictor signals acquirer to run.
//   3. Acquirer marks victim in-flight (acquireSessionSlot equivalent).
//   4. Barrier phase 1→2: acquirer signals evictor.
//   5. Evictor calls take — victim is now in-flight but is taken anyway.
bool buggyEvictWithBarrier(tt::utils::ConcurrentMap<int, Session>& sessions,
                           int victimKey, Barrier& barrier) {
  // Step 1: scan (identical to forEach in evictOldSessions).
  std::vector<int> candidates;
  sessions.forEach([&](const int& key, Session& s) {
    if (!s.in_flight) candidates.push_back(key);
  });
  // ↑ Lock released. TOCTOU window opens here.

  barrier.advance();  // signal: "scan done, acquirer may run"
  barrier.waitFor(2);  // wait for acquirer to mark session in-flight

  // Step 2: take unconditionally — THE BUG.
  for (int key : candidates) {
    if (key == victimKey) {
      auto s = sessions.take(key);
      return s.has_value();
    }
  }
  return false;
}

// ---------------------------------------------------------------------------

TEST(BugRepro, OriginalTakeEvictsInFlightSessionEveryTime) {
  constexpr int kNumSessions = 8;
  constexpr int kVictim = 3;
  constexpr int kIterations = 200;

  int wrongEvictions = 0;

  for (int i = 0; i < kIterations; ++i) {
    tt::utils::ConcurrentMap<int, Session> sessions;
    for (int k = 0; k < kNumSessions; ++k) {
      sessions.insert(k, Session{false, k});
    }

    Barrier barrier;
    bool evictedVictim = false;

    // Evictor thread: scan then take (buggy path).
    std::thread evictor([&] {
      evictedVictim = buggyEvictWithBarrier(sessions, kVictim, barrier);
    });

    // Main thread: wait for scan, then mark victim in-flight (acquireSessionSlot).
    barrier.waitFor(1);
    sessions.modify(kVictim, [](Session& s) { s.in_flight = true; });
    barrier.advance();  // signal evictor to continue with take

    evictor.join();

    if (evictedVictim) ++wrongEvictions;

    // Restore for next iteration.
    if (!sessions.contains(kVictim))
      sessions.insert(kVictim, Session{false, kVictim});
  }

  std::cout << "[bug]  wrong_evictions=" << wrongEvictions << "/" << kIterations
            << "\n";
  EXPECT_EQ(wrongEvictions, kIterations)
      << "Original take() must evict the in-flight session on every iteration "
         "when the race window is forced open";
}

}  // namespace
