// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <gtest/gtest.h>

namespace tt::gateway {
namespace {

PrefillSnapshot prefill(std::string id, bool healthy = true,
                        uint32_t inFlight = 0, uint32_t maxInFlight = 0) {
  PrefillSnapshot p;
  p.server_id = std::move(id);
  p.healthy = healthy;
  p.in_flight = inFlight;
  p.max_in_flight = maxInFlight;
  return p;
}

TEST(PrefillSelectorTest, NoPeersAvailableWhenAllDown) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", false),
                                           prefill("B", false)};
  size_t cursor = 0;
  auto chosen = selectPrefill(prefills, /*hash=*/0, std::nullopt, cursor);
  EXPECT_FALSE(chosen.has_value());
}

TEST(PrefillSelectorTest, NoPeersAvailableWhenAllAtMaxInFlight) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, /*inFlight=*/4, /*maxInFlight=*/4),
      prefill("B", true, /*inFlight=*/8, /*maxInFlight=*/8)};
  size_t cursor = 0;
  auto chosen = selectPrefill(prefills, 0, std::nullopt, cursor);
  EXPECT_FALSE(chosen.has_value());
}

TEST(PrefillSelectorTest, EqualityMatchPicksStickyTarget) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, /*inFlight=*/5),
      prefill("B", true, /*inFlight=*/0)};
  size_t cursor = 0;
  // Sticky -> A even though B is less loaded.
  auto chosen = selectPrefill(prefills, /*hash=*/42, std::string{"A"}, cursor);
  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "A");
}

TEST(PrefillSelectorTest, StickyFallsBackToLeastLoadedWhenTargetUnhealthy) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", /*healthy=*/false),
                                           prefill("B", true, 0)};
  size_t cursor = 0;
  auto chosen = selectPrefill(prefills, 42, std::string{"A"}, cursor);
  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

TEST(PrefillSelectorTest, StickyFallsBackToLeastLoadedWhenTargetOverloaded) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, /*inFlight=*/8, /*maxInFlight=*/8),
      prefill("B", true, /*inFlight=*/2, /*maxInFlight=*/8)};
  size_t cursor = 0;
  auto chosen = selectPrefill(prefills, 42, std::string{"A"}, cursor);
  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

TEST(PrefillSelectorTest, LeastInflightWinsOverLoaded) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, 3),
                                           prefill("B", true, 1),
                                           prefill("C", true, 5)};
  size_t cursor = 0;
  auto chosen = selectPrefill(prefills, 0, std::nullopt, cursor);
  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

TEST(PrefillSelectorTest, RoundRobinAmongTiedLeastLoaded) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, 0),
                                           prefill("B", true, 0),
                                           prefill("C", true, 0)};
  size_t cursor = 0;
  auto c1 = selectPrefill(prefills, 0, std::nullopt, cursor);
  auto c2 = selectPrefill(prefills, 0, std::nullopt, cursor);
  auto c3 = selectPrefill(prefills, 0, std::nullopt, cursor);
  auto c4 = selectPrefill(prefills, 0, std::nullopt, cursor);

  // Cursor advances each call -> A, B, C, then wraps to A.
  ASSERT_TRUE(c1 && c2 && c3 && c4);
  EXPECT_EQ(*c1, "A");
  EXPECT_EQ(*c2, "B");
  EXPECT_EQ(*c3, "C");
  EXPECT_EQ(*c4, "A");
}

TEST(PrefillSelectorTest, HashOfZeroIgnoresSticky) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, 5),
                                           prefill("B", true, 0)};
  size_t cursor = 0;
  auto chosen = selectPrefill(prefills, /*hash=*/0, std::string{"A"}, cursor);
  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

}  // namespace
}  // namespace tt::gateway
