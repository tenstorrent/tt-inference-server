// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_registry.hpp"

#include <gtest/gtest.h>

#include <string>

namespace tt::gateway {
namespace {

TEST(PrefillRegistryTest, PreRegisteredPrefillIsUnhealthyUntilMarked) {
  PrefillRegistry reg;
  reg.preRegister("A", /*manager=*/nullptr);

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].server_id, "A");
  EXPECT_FALSE(snaps[0].healthy);
}

TEST(PrefillRegistryTest, MarkRegisteredTurnsPrefillHealthy) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);

  ASSERT_TRUE(reg.markRegistered("A", /*max_in_flight=*/8));

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_TRUE(snaps[0].healthy);
  EXPECT_EQ(snaps[0].max_in_flight, 8u);
}

TEST(PrefillRegistryTest, SetAcceptingTasksUpdatesSnapshot) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.markRegistered("A", 4);

  reg.setAcceptingTasks("A", false);

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_FALSE(snaps[0].accepting_tasks);

  reg.setAcceptingTasks("A", true);
  snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_TRUE(snaps[0].accepting_tasks);
}

TEST(PrefillRegistryTest, MarkRegisteredReturnsFalseForUnknownPrefill) {
  PrefillRegistry reg;
  EXPECT_FALSE(reg.markRegistered("UNKNOWN", 4));
}

TEST(PrefillRegistryTest, MarkDownTurnsPrefillUnhealthyAndFiresCallback) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.markRegistered("A", 4);

  std::string seenId;
  reg.setOnPrefillDown([&](const std::string& id) { seenId = id; });

  reg.markDown("A");

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_FALSE(snaps[0].healthy);
  EXPECT_EQ(seenId, "A");
}

TEST(PrefillRegistryTest, MarkDownIsNoopForUnknownPrefill) {
  PrefillRegistry reg;
  bool fired = false;
  reg.setOnPrefillDown([&](const std::string&) { fired = true; });

  reg.markDown("UNKNOWN");
  EXPECT_FALSE(fired);
}

TEST(PrefillRegistryTest, InflightCountSaturatesAtZeroOnDecrement) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);

  reg.decrementInflight("A");
  reg.decrementInflight("A");

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].in_flight, 0u);
}

TEST(PrefillRegistryTest, IncrementDecrementInflightUpdatesSnapshot) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);

  reg.incrementInflight("A");
  reg.incrementInflight("A");
  reg.incrementInflight("A");
  reg.decrementInflight("A");

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].in_flight, 2u);
}

TEST(PrefillRegistryTest, GetSocketManagerReturnsNullptrForUnknown) {
  PrefillRegistry reg;
  EXPECT_EQ(reg.getSocketManager("UNKNOWN"), nullptr);
}

TEST(PrefillRegistryTest, CacheBlockDeltasAreTrackedPerPrefill) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.preRegister("B", nullptr);

  reg.addCachedBlocks("A", {1, 2, 3});
  reg.addCachedBlocks("B", {1, 4});
  reg.evictCachedBlocks("A", {2});

  reg.addCachedBlocks("UNKNOWN", {7});
  reg.evictCachedBlocks("UNKNOWN", {7});

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 2u);
  for (const auto& snap : snaps) {
    if (snap.server_id == "A") {
      EXPECT_EQ(snap.cached_blocks, 2u);
    } else if (snap.server_id == "B") {
      EXPECT_EQ(snap.cached_blocks, 2u);
    } else {
      FAIL() << "Unexpected server id " << snap.server_id;
    }
  }
}

TEST(PrefillRegistryTest, RoutingSnapshotComputesContiguousPrefixDepth) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.preRegister("B", nullptr);
  reg.markRegistered("A", 4);
  reg.markRegistered("B", 4);
  reg.addCachedBlocks("A", {10, 30});
  reg.addCachedBlocks("B", {10, 20});

  auto snaps = reg.routingSnapshot({10, 20, 30});

  ASSERT_EQ(snaps.size(), 2u);
  for (const auto& snap : snaps) {
    if (snap.server_id == "A") {
      EXPECT_EQ(snap.prefix_match_depth, 1u);
    } else if (snap.server_id == "B") {
      EXPECT_EQ(snap.prefix_match_depth, 2u);
    } else {
      FAIL() << "Unexpected server id " << snap.server_id;
    }
  }
}

TEST(PrefillRegistryTest, RoutingSnapshotReflectsEvictionsAndMarkDown) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.markRegistered("A", 4);
  reg.addCachedBlocks("A", {10, 20});

  auto snaps = reg.routingSnapshot({10, 20});
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].prefix_match_depth, 2u);

  reg.evictCachedBlocks("A", {20});
  snaps = reg.routingSnapshot({10, 20});
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].prefix_match_depth, 1u);

  reg.markDown("A");
  snaps = reg.routingSnapshot({10});
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].prefix_match_depth, 0u);
}

}  // namespace
}  // namespace tt::gateway
