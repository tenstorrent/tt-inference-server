// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Deterministic reproducer for the TOCTOU race in evictOldSessions
// (issues #2907 / #3001).
//
// Uses a two-phase barrier to reliably open the race window:
//   Phase 1 – evictor thread calls forEach (scan) and PAUSES.
//   Phase 2 – acquirer thread marks session as in-flight, then signals done.
//   Phase 3 – evictor thread resumes and calls take/takeIf.
//
// With the BUG (take):  the in-flight session is taken → wrong_eviction.
// With the FIX (takeIf): takeIf sees in_flight=true → skips → 0 wrong.

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

// Two-barrier synchroniser to force the exact TOCTOU window open every time.
struct TwoPhaseBarrier {
  std::mutex mu;
  std::condition_variable cv;
  // States: 0 = initial, 1 = scan done (acquirer can run), 2 = acquire done
  int phase{0};

  void waitForPhase(int p) {
    std::unique_lock lk(mu);
    cv.wait(lk, [&] { return phase >= p; });
  }

  void advance() {
    std::lock_guard lk(mu);
    ++phase;
    cv.notify_all();
  }

  void reset() {
    std::lock_guard lk(mu);
    phase = 0;
  }
};

// ---- Buggy eviction with barrier (mirrors the old take-without-recheck) ---

bool evictBuggyWithBarrier(tt::utils::ConcurrentMap<int, Session>& sessions,
                           int victimKey, TwoPhaseBarrier& barrier) {
  std::vector<int> candidates;
  sessions.forEach([&](const int& key, Session& s) {
    if (!s.in_flight) candidates.push_back(key);
  });
  // ↑ Lock released after forEach.

  // Signal acquirer: "scan done, please mark in-flight now"
  barrier.advance();  // phase 0→1
  // Wait for acquirer to finish marking in-flight.
  barrier.waitForPhase(2);

  // Now take unconditionally — THE BUG: doesn't re-check in_flight.
  for (int key : candidates) {
    if (key == victimKey) {
      auto s = sessions.take(key);
      return s.has_value();  // true = wrongly evicted an in-flight session
    }
  }
  return false;
}

// ---- Fixed eviction with barrier (uses takeIf) ----------------------------

bool evictFixedWithBarrier(tt::utils::ConcurrentMap<int, Session>& sessions,
                           int victimKey, TwoPhaseBarrier& barrier) {
  std::vector<int> candidates;
  sessions.forEach([&](const int& key, Session& s) {
    if (!s.in_flight) candidates.push_back(key);
  });

  barrier.advance();  // phase 0→1
  barrier.waitForPhase(2);

  for (int key : candidates) {
    if (key == victimKey) {
      // Atomically check-and-remove: only take if still not in-flight.
      auto s =
          sessions.takeIf(key, [](const Session& s) { return !s.in_flight; });
      return s.has_value();  // false = correctly skipped in-flight session
    }
  }
  return false;
}

// ---- Harness ---------------------------------------------------------------

template <typename EvictFn>
int countWrongEvictions(EvictFn evictFn, int iterations = 200) {
  constexpr int kMaxSessions = 8;
  constexpr int kVictimKey = 3;

  int wrong = 0;
  for (int iter = 0; iter < iterations; ++iter) {
    tt::utils::ConcurrentMap<int, Session> sessions;
    for (int i = 0; i < kMaxSessions; ++i) {
      sessions.insert(i, Session{false, i});
    }

    TwoPhaseBarrier barrier;
    bool evictedVictim = false;

    // Thread A: evict (with barrier to force the TOCTOU window)
    std::thread evictor([&] {
      evictedVictim = evictFn(sessions, kVictimKey, barrier);
    });

    // Thread B (this thread): wait for scan to finish, then mark victim
    // in-flight (simulates acquireSessionSlot)
    barrier.waitForPhase(1);
    sessions.modify(kVictimKey, [](Session& s) { s.in_flight = true; });
    barrier.advance();  // phase 1→2 (signal evictor to continue)

    evictor.join();

    if (evictedVictim) ++wrong;

    // Restore for next iteration
    if (!sessions.contains(kVictimKey))
      sessions.insert(kVictimKey, Session{false, kVictimKey});
  }
  return wrong;
}

// ---------------------------------------------------------------------------

TEST(ToctouRaceRepro, BuggyCodeEvictsInFlightSessionsEveryTime) {
  int wrong = countWrongEvictions(evictBuggyWithBarrier);
  std::cout << "[buggy]  wrong_evictions=" << wrong << "/200\n";
  // With the forced race window, the bug fires on every single iteration.
  EXPECT_GT(wrong, 0)
      << "Expected the buggy code to evict in-flight sessions when the "
         "TOCTOU window is forced open";
}

TEST(ToctouRaceRepro, FixedCodeNeverEvictsInFlightSessions) {
  int wrong = countWrongEvictions(evictFixedWithBarrier);
  std::cout << "[fixed]  wrong_evictions=" << wrong << "/200\n";
  EXPECT_EQ(wrong, 0)
      << "takeIf must atomically skip in-flight sessions even when the "
         "TOCTOU window is fully open";
}

}  // namespace
