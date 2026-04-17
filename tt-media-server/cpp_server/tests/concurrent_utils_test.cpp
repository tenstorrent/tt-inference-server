// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "utils/concurrent_map.hpp"
#include "utils/concurrent_queue.hpp"

namespace {

// ---------------------------------------------------------------------------
// ConcurrentMap::takeIf tests
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
  // The entry must still be in the map.
  EXPECT_TRUE(map.contains(1));
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
// as "in-flight" (predicate false) while another tries to evict them.
// With takeIf the eviction must never remove an item that was concurrently
// marked in-flight.
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

  // Thread 1: mark items in-flight
  std::thread marker([&] {
    for (int i = 0; i < kNumItems; ++i) {
      map.modify(i, [](Item& item) { item.in_flight = true; });
    }
  });

  // Thread 2: evict items that are NOT in-flight
  std::thread evictor([&] {
    for (int i = 0; i < kNumItems; ++i) {
      auto taken =
          map.takeIf(i, [](const Item& item) { return !item.in_flight; });
      if (taken.has_value() && taken->in_flight) {
        // takeIf returned an item whose predicate should have been false
        ++falseEvictions;
      }
    }
  });

  marker.join();
  evictor.join();

  EXPECT_EQ(falseEvictions.load(), 0)
      << "takeIf evicted an in-flight item — TOCTOU race not fixed";
}

// ---------------------------------------------------------------------------
// ConcurrentQueue move overload tests
// ---------------------------------------------------------------------------

TEST(ConcurrentQueueMoveOverload, MovedValueArrivesInQueue) {
  tt::utils::ConcurrentQueue<std::string> q;
  std::string s = "hello";
  q.push(std::move(s));
  // After a genuine move, s should be empty (moved-from).
  EXPECT_TRUE(s.empty());
  auto items = q.drain();
  ASSERT_EQ(items.size(), 1u);
  EXPECT_EQ(items[0], "hello");
}

TEST(ConcurrentQueueMoveOverload, MoveOverloadUsedForRvalues) {
  // Use a move-only type to confirm the move overload is selected.
  tt::utils::ConcurrentQueue<std::unique_ptr<int>> q;
  q.push(std::make_unique<int>(42));
  auto items = q.drain();
  ASSERT_EQ(items.size(), 1u);
  ASSERT_NE(items[0], nullptr);
  EXPECT_EQ(*items[0], 42);
}

}  // namespace
