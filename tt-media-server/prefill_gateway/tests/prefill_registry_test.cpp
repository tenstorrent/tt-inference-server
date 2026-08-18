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
  EXPECT_EQ(snaps[0].serverId, "A");
  EXPECT_FALSE(snaps[0].healthy);
}

TEST(PrefillRegistryTest, MarkRegisteredTurnsPrefillHealthy) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);

  ASSERT_TRUE(reg.markRegistered("A", /*max_in_flight=*/8));

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_TRUE(snaps[0].healthy);
  EXPECT_EQ(snaps[0].maxInFlight, 8u);
}

TEST(PrefillRegistryTest, SetAcceptingTasksUpdatesSnapshot) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.markRegistered("A", 4);

  reg.setAcceptingTasks("A", false);

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_FALSE(snaps[0].acceptingTasks);

  reg.setAcceptingTasks("A", true);
  snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_TRUE(snaps[0].acceptingTasks);
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
  EXPECT_EQ(snaps[0].inFlight, 0u);
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
  EXPECT_EQ(snaps[0].inFlight, 2u);
}

TEST(PrefillRegistryTest, GetSocketManagerReturnsNullptrForUnknown) {
  PrefillRegistry reg;
  EXPECT_EQ(reg.getSocketManager("UNKNOWN"), nullptr);
}

TEST(PrefillRegistryTest, CacheBlocksAreTrackedPerPrefill) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.preRegister("B", nullptr);

  reg.addCachedBlocks("A", {1, 2, 3});
  reg.addCachedBlocks("B", {1, 4});

  reg.addCachedBlocks("UNKNOWN", {7});

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 2u);
  for (const auto& snap : snaps) {
    if (snap.serverId == "A") {
      EXPECT_EQ(snap.cachedBlocks, 3u);
    } else if (snap.serverId == "B") {
      EXPECT_EQ(snap.cachedBlocks, 2u);
    } else {
      FAIL() << "Unexpected server id " << snap.serverId;
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
    if (snap.serverId == "A") {
      EXPECT_EQ(snap.prefixMatchDepth, 1u);
    } else if (snap.serverId == "B") {
      EXPECT_EQ(snap.prefixMatchDepth, 2u);
    } else {
      FAIL() << "Unexpected server id " << snap.serverId;
    }
  }
}

TEST(PrefillRegistryTest, MarkDownClearsRoutingCacheView) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.markRegistered("A", 4);
  reg.addCachedBlocks("A", {10, 20});

  auto snaps = reg.routingSnapshot({10, 20});
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].prefixMatchDepth, 2u);

  reg.markDown("A");
  snaps = reg.routingSnapshot({10});
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].prefixMatchDepth, 0u);
}

}  // namespace
}  // namespace tt::gateway
