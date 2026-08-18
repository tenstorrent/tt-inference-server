// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/concurrent_map.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "utils/concurrent_queue.hpp"

namespace {

// ---------------------------------------------------------------------------
// ConcurrentMap::takeIf — unit tests
// ---------------------------------------------------------------------------

TEST(ConcurrentMapTakeIf, ReturnsNulloptWhenKeyAbsent) {
  tt::utils::ConcurrentMap<int, std::string> map;
  auto result = map.takeIf(42, [](const std::string&) { return true; });
  EXPECT_FALSE(result.has_value());
}

TEST(ConcurrentMapTakeIf, ReturnsNulloptWhenPredicateFalse) {
  tt::utils::ConcurrentMap<int, std::string> map;
  map.insert(1, "hello");
  auto result = map.takeIf(1, [](const std::string&) { return false; });
  EXPECT_FALSE(result.has_value());
  EXPECT_TRUE(map.contains(1));  // entry must remain in the map
}

TEST(ConcurrentMapTakeIf, TakesAndRemovesWhenPredicateTrue) {
  tt::utils::ConcurrentMap<int, std::string> map;
  map.insert(1, "hello");
  auto result = map.takeIf(1, [](const std::string&) { return true; });
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "hello");
  EXPECT_FALSE(map.contains(1));
}

TEST(ConcurrentMapTakeIf, PredicateReceivesCorrectValue) {
  tt::utils::ConcurrentMap<int, int> map;
  map.insert(1, 99);
  std::optional<int> seenValue;
  map.takeIf(1, [&seenValue](const int& v) {
    seenValue = v;
    return false;
  });
  ASSERT_TRUE(seenValue.has_value());
  EXPECT_EQ(*seenValue, 99);
}

// Simulates the eviction TOCTOU race: one thread races to mark items
// in-flight while another tries to evict them. takeIf must never remove an
// item whose predicate was made false by a concurrent modifier.
TEST(ConcurrentMapTakeIf, NeverEvictsInFlightItemUnderContention) {
  struct Item {
    bool in_flight{false};
  };

  tt::utils::ConcurrentMap<int, Item> map;
  constexpr int kNumItems = 1000;
  for (int i = 0; i < kNumItems; ++i) {
    map.insert(i, Item{});
  }

  std::atomic<int> falseEvictions{0};

  std::thread marker([&] {
    for (int i = 0; i < kNumItems; ++i) {
      map.modify(i, [](Item& item) { item.in_flight = true; });
    }
  });

  std::thread evictor([&] {
    for (int i = 0; i < kNumItems; ++i) {
      auto taken =
          map.takeIf(i, [](const Item& item) { return !item.in_flight; });
      if (taken.has_value() && taken->in_flight) ++falseEvictions;
    }
  });

  marker.join();
  evictor.join();

  EXPECT_EQ(falseEvictions.load(), 0)
      << "takeIf evicted an in-flight item — TOCTOU race not fixed";
}

// ---------------------------------------------------------------------------
// ConcurrentQueue — move overload tests
// ---------------------------------------------------------------------------

TEST(ConcurrentQueueMoveOverload, MovedValueArrivesInQueue) {
  tt::utils::ConcurrentQueue<std::string> q;
  std::string s = "hello";
  q.push(std::move(s));
  EXPECT_TRUE(s.empty());  // must have been genuinely moved
  auto items = q.drain();
  ASSERT_EQ(items.size(), 1u);
  EXPECT_EQ(items[0], "hello");
}

TEST(ConcurrentQueueMoveOverload, MoveOverloadUsedForRvalues) {
  tt::utils::ConcurrentQueue<std::unique_ptr<int>> q;
  q.push(std::make_unique<int>(42));
  auto items = q.drain();
  ASSERT_EQ(items.size(), 1u);
  ASSERT_NE(items[0], nullptr);
  EXPECT_EQ(*items[0], 42);
}

// ---------------------------------------------------------------------------
// TOCTOU race — barrier-driven regression tests
//
// Two-phase barrier forces the exact race window open on every iteration:
//   Phase 1 – evictor's forEach scan completes (lock released).
//   Phase 2 – acquirer marks victim in-flight, signals done.
//   Phase 3 – evictor calls take/takeIf.
//
// With take  (buggy):  victim is evicted even though it's now in-flight.
// With takeIf (fixed): victim is skipped because predicate re-checks state.
// ---------------------------------------------------------------------------

struct Session {
  bool in_flight{false};
  int slot_id{-1};
};

struct TwoPhaseBarrier {
  std::mutex mu;
  std::condition_variable cv;
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

bool evictBuggy(tt::utils::ConcurrentMap<int, Session>& sessions, int victim,
                TwoPhaseBarrier& barrier) {
  std::vector<int> candidates;
  sessions.forEach([&](const int& key, Session& s) {
    if (!s.in_flight) candidates.push_back(key);
  });
  // Lock released — TOCTOU window is now open.
  barrier.advance();        // signal acquirer: scan is done
  barrier.waitForPhase(2);  // wait for acquirer to mark in-flight

  for (int key : candidates) {
    if (key == victim) return sessions.take(key).has_value();  // THE BUG
  }
  return false;
}

bool evictFixed(tt::utils::ConcurrentMap<int, Session>& sessions, int victim,
                TwoPhaseBarrier& barrier) {
  std::vector<int> candidates;
  sessions.forEach([&](const int& key, Session& s) {
    if (!s.in_flight) candidates.push_back(key);
  });
  barrier.advance();
  barrier.waitForPhase(2);

  for (int key : candidates) {
    if (key == victim) {
      // Atomically re-check predicate inside the lock — THE FIX.
      return sessions.takeIf(key, [](const Session& s) { return !s.in_flight; })
          .has_value();
    }
  }
  return false;
}

template <typename EvictFn>
int countWrongEvictions(EvictFn evictFn, int iterations = 200) {
  constexpr int kNumSessions = 8;
  constexpr int kVictim = 3;
  int wrong = 0;

  for (int iter = 0; iter < iterations; ++iter) {
    tt::utils::ConcurrentMap<int, Session> sessions;
    for (int i = 0; i < kNumSessions; ++i) {
      sessions.insert(i, Session{false, i});
    }

    TwoPhaseBarrier barrier;
    bool evictedVictim = false;

    std::thread evictor(
        [&] { evictedVictim = evictFn(sessions, kVictim, barrier); });

    barrier.waitForPhase(1);
    sessions.modify(kVictim, [](Session& s) { s.in_flight = true; });
    barrier.advance();  // phase 1→2

    evictor.join();
    if (evictedVictim) ++wrong;

    if (!sessions.contains(kVictim))
      sessions.insert(kVictim, Session{false, kVictim});
  }
  return wrong;
}

TEST(ToctouRaceRepro, BuggyTakeEvictsInFlightSessionOnEveryIteration) {
  constexpr int kIterations = 200;
  int wrong = countWrongEvictions(evictBuggy, kIterations);
  std::cout << "[buggy]  wrong_evictions=" << wrong << "/" << kIterations
            << "\n";
  // The barrier forces the race window open deterministically, so the bug
  // must fire on every single iteration.
  EXPECT_EQ(wrong, kIterations)
      << "take() must evict the in-flight session on every iteration when the "
         "TOCTOU window is forced open";
}

TEST(ToctouRaceRepro, FixedTakeIfNeverEvictsInFlightSession) {
  constexpr int kIterations = 200;
  int wrong = countWrongEvictions(evictFixed, kIterations);
  std::cout << "[fixed]  wrong_evictions=" << wrong << "/" << kIterations
            << "\n";
  EXPECT_EQ(wrong, 0)
      << "takeIf must atomically skip in-flight sessions even when the "
         "TOCTOU window is fully open";
}

}  // namespace
