// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <gtest/gtest.h>

namespace tt::gateway {
namespace {

PrefillSnapshot prefill(std::string id, bool healthy = true,
                        uint32_t inFlight = 0, uint32_t maxInFlight = 0) {
  PrefillSnapshot p;
  p.serverId = std::move(id);
  p.healthy = healthy;
  p.inFlight = inFlight;
  p.maxInFlight = maxInFlight;
  return p;
}

PrefillSnapshot prefixMatchedPrefill(std::string id, size_t prefixMatchDepth,
                                     uint32_t inFlight = 0) {
  auto p = prefill(std::move(id), true, inFlight);
  p.prefixMatchDepth = prefixMatchDepth;
  return p;
}

std::optional<std::string> selectedServer(
    const std::vector<PrefillSnapshot>& prefills, size_t& roundRobinCursor,
    std::optional<std::string> preferredPrefillId = std::nullopt) {
  return selectPrefill(prefills, roundRobinCursor, preferredPrefillId).serverId;
}

TEST(PrefillSelectorTest, NoPeersAvailableWhenAllDown) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", false),
                                           prefill("B", false)};
  size_t cursor = 0;
  auto chosen = selectedServer(prefills, cursor);
  EXPECT_FALSE(chosen.has_value());
}

TEST(PrefillSelectorTest, NoPeersAvailableWhenAllAtMaxInFlight) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, /*inFlight=*/4, /*maxInFlight=*/4),
      prefill("B", true, /*inFlight=*/8, /*maxInFlight=*/8)};
  size_t cursor = 0;
  auto chosen = selectedServer(prefills, cursor);
  EXPECT_FALSE(chosen.has_value());
}

TEST(PrefillSelectorTest, NoPeersAvailableWhenNotAcceptingTasks) {
  auto disabled = prefill("A", true);
  disabled.acceptingTasks = false;
  std::vector<PrefillSnapshot> prefills = {disabled};
  size_t cursor = 0;

  auto chosen = selectedServer(prefills, cursor);

  EXPECT_FALSE(chosen.has_value());
}

TEST(PrefillSelectorTest, SummarizesEligibilityReasons) {
  auto disabled = prefill("B", true);
  disabled.acceptingTasks = false;
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", /*healthy=*/false),
      disabled,
      prefill("C", true, /*inFlight=*/4, /*maxInFlight=*/4),
      prefill("D", true, /*inFlight=*/1, /*maxInFlight=*/4),
  };

  const auto summary = summarizePrefillEligibility(prefills);

  EXPECT_EQ(summary.total, 4u);
  EXPECT_EQ(summary.healthy, 3u);
  EXPECT_EQ(summary.accepting, 2u);
  EXPECT_EQ(summary.capacityAvailable, 1u);
}

TEST(PrefillSelectorTest, LongestPrefixMatchWinsOverLowerLoad) {
  std::vector<PrefillSnapshot> prefills = {
      prefixMatchedPrefill("A", /*prefixMatchDepth=*/3, /*inFlight=*/5),
      prefixMatchedPrefill("B", /*prefixMatchDepth=*/2, /*inFlight=*/0)};
  size_t cursor = 0;

  auto chosen = selectedServer(prefills, cursor);

  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "A");
}

TEST(PrefillSelectorTest, PreferredPrefillWinsWhenEligible) {
  std::vector<PrefillSnapshot> prefills = {
      prefixMatchedPrefill("A", /*prefixMatchDepth=*/3, /*inFlight=*/0),
      prefixMatchedPrefill("B", /*prefixMatchDepth=*/0, /*inFlight=*/3)};
  size_t cursor = 0;

  const auto selection = selectPrefill(prefills, cursor, std::string("B"));

  ASSERT_TRUE(selection.serverId.has_value());
  EXPECT_EQ(*selection.serverId, "B");
  EXPECT_EQ(selection.reason, PrefillRoutingReason::PreferredPrefill);
  EXPECT_EQ(selection.prefixMatchDepth, 0u);
}

TEST(PrefillSelectorTest, IneligiblePreferredPrefillFallsBack) {
  auto preferred = prefixMatchedPrefill("A", /*prefixMatchDepth=*/3);
  preferred.acceptingTasks = false;
  std::vector<PrefillSnapshot> prefills = {
      preferred, prefixMatchedPrefill("B", /*prefixMatchDepth=*/1)};
  size_t cursor = 0;

  const auto selection = selectPrefill(prefills, cursor, std::string("A"));

  ASSERT_TRUE(selection.serverId.has_value());
  EXPECT_EQ(*selection.serverId, "B");
  EXPECT_EQ(selection.reason, PrefillRoutingReason::PrefixMatch);
}

TEST(PrefillSelectorTest, PrefixMatchIgnoresIneligiblePrefills) {
  auto disabled = prefixMatchedPrefill("A", /*prefixMatchDepth=*/3);
  disabled.acceptingTasks = false;
  std::vector<PrefillSnapshot> prefills = {
      disabled, prefixMatchedPrefill("B", /*prefixMatchDepth=*/1)};
  size_t cursor = 0;

  auto chosen = selectedServer(prefills, cursor);

  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

TEST(PrefillSelectorTest, PrefixTieBreaksByLeastLoaded) {
  std::vector<PrefillSnapshot> prefills = {
      prefixMatchedPrefill("A", /*prefixMatchDepth=*/2, /*inFlight=*/4),
      prefixMatchedPrefill("B", /*prefixMatchDepth=*/2, /*inFlight=*/1)};
  size_t cursor = 0;

  auto chosen = selectedServer(prefills, cursor);

  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

TEST(PrefillSelectorTest, PrefixTieBreaksByRoundRobinWhenEquallyLoaded) {
  std::vector<PrefillSnapshot> prefills = {
      prefixMatchedPrefill("A", /*prefixMatchDepth=*/2, /*inFlight=*/0),
      prefixMatchedPrefill("B", /*prefixMatchDepth=*/2, /*inFlight=*/0)};
  size_t cursor = 0;
  auto c1 = selectedServer(prefills, cursor);
  auto c2 = selectedServer(prefills, cursor);

  ASSERT_TRUE(c1 && c2);
  EXPECT_EQ(*c1, "A");
  EXPECT_EQ(*c2, "B");
}

TEST(PrefillSelectorTest, LeastInflightWinsOverLoaded) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, 3), prefill("B", true, 1), prefill("C", true, 5)};
  size_t cursor = 0;
  auto chosen = selectedServer(prefills, cursor);
  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

TEST(PrefillSelectorTest, RoundRobinAmongTiedLeastLoaded) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, 0), prefill("B", true, 0), prefill("C", true, 0)};
  size_t cursor = 0;
  auto c1 = selectedServer(prefills, cursor);
  auto c2 = selectedServer(prefills, cursor);
  auto c3 = selectedServer(prefills, cursor);
  auto c4 = selectedServer(prefills, cursor);

  // Cursor advances each call -> A, B, C, then wraps to A.
  ASSERT_TRUE(c1 && c2 && c3 && c4);
  EXPECT_EQ(*c1, "A");
  EXPECT_EQ(*c2, "B");
  EXPECT_EQ(*c3, "C");
  EXPECT_EQ(*c4, "A");
}

TEST(PrefillSelectorTest, NoPrefixMatchFallsBackToLeastLoaded) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, 5),
                                           prefill("B", true, 0)};
  size_t cursor = 0;
  auto chosen = selectedServer(prefills, cursor);
  ASSERT_TRUE(chosen.has_value());
  EXPECT_EQ(*chosen, "B");
}

}  // namespace
}  // namespace tt::gateway
