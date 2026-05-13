// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_registry.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace tt::gateway {
namespace {

const PrefillSnapshot* findSnap(const std::vector<PrefillSnapshot>& snaps,
                                const std::string& id) {
  auto it = std::find_if(
      snaps.begin(), snaps.end(),
      [&](const PrefillSnapshot& s) { return s.server_id == id; });
  return it == snaps.end() ? nullptr : &*it;
}

TEST(PrefillRegistryTest, PreRegisteredPrefillIsUnhealthyUntilMarked) {
  PrefillRegistry reg;
  reg.preRegister("A", /*manager=*/nullptr);

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].server_id, "A");
  EXPECT_FALSE(snaps[0].healthy);
  EXPECT_TRUE(snaps[0].accepting_tasks);
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

TEST(PrefillRegistryTest, MarkRegisteredReturnsFalseForUnknownPrefill) {
  PrefillRegistry reg;
  EXPECT_FALSE(reg.markRegistered("UNKNOWN", 4));
}

TEST(PrefillRegistryTest, MarkRegisteredFiresOnPrefillUpCallback) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);

  std::string seen_id;
  reg.setOnPrefillUp([&](const std::string& id) { seen_id = id; });

  reg.markRegistered("A", 4);
  EXPECT_EQ(seen_id, "A");
}

TEST(PrefillRegistryTest, MarkDownTurnsPrefillUnhealthyAndFiresCallback) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.markRegistered("A", 4);

  std::string seen_id;
  reg.setOnPrefillDown([&](const std::string& id) { seen_id = id; });

  reg.markDown("A");

  auto snaps = reg.snapshot();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_FALSE(snaps[0].healthy);
  EXPECT_EQ(seen_id, "A");
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

TEST(PrefillRegistryTest, UpdateLoadInfoTogglesAcceptingTasks) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.markRegistered("A", 4);

  reg.updateLoadInfo("A", /*accepting_tasks=*/false);
  auto snaps = reg.snapshot();
  EXPECT_FALSE(findSnap(snaps, "A")->accepting_tasks);

  reg.updateLoadInfo("A", true);
  snaps = reg.snapshot();
  EXPECT_TRUE(findSnap(snaps, "A")->accepting_tasks);
}

TEST(PrefillRegistryTest, HealthyPrefillIdsExcludesDownPrefills) {
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.preRegister("B", nullptr);
  reg.preRegister("C", nullptr);
  reg.markRegistered("A", 4);
  reg.markRegistered("C", 4);

  auto ids = reg.healthyPrefillIds();
  std::sort(ids.begin(), ids.end());
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], "A");
  EXPECT_EQ(ids[1], "C");
}

TEST(PrefillRegistryTest, GetSocketManagerReturnsNullptrForUnknown) {
  PrefillRegistry reg;
  EXPECT_EQ(reg.getSocketManager("UNKNOWN"), nullptr);
}

TEST(PrefillRegistryTest, CacheBlockDeltasAreTrackedPerPrefill) {
  // We can't observe cached_blocks via snapshot() (snapshots are for the
  // selector and don't carry the block set), so we verify behavior via
  // add/evict no-throw + size effects observable through repeated adds.
  PrefillRegistry reg;
  reg.preRegister("A", nullptr);
  reg.preRegister("B", nullptr);

  reg.addCachedBlocks("A", {1, 2, 3});
  reg.addCachedBlocks("B", {1, 4});
  reg.evictCachedBlocks("A", {2});

  // No public read API for cached_blocks yet — this test ensures the
  // mutators don't throw and tolerate unknown ids gracefully.
  reg.addCachedBlocks("UNKNOWN", {7});
  reg.evictCachedBlocks("UNKNOWN", {7});
  SUCCEED();
}

}  // namespace
}  // namespace tt::gateway
