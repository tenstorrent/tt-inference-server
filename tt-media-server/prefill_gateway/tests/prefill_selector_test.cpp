// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <gtest/gtest.h>

namespace tt::gateway {
namespace {

PrefillSnapshot prefill(std::string id, bool healthy = true,
                        bool accepting = true, uint32_t inFlight = 0,
                        uint32_t maxInFlight = 0) {
  PrefillSnapshot p;
  p.server_id = std::move(id);
  p.healthy = healthy;
  p.accepting_tasks = accepting;
  p.in_flight = inFlight;
  p.max_in_flight = maxInFlight;
  return p;
}

TEST(PrefillSelectorTest, NoPeersAvailableWhenAllDown) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", false),
                                           prefill("B", false)};
  size_t cursor = 0;
  SelectionResult d = selectPrefill(prefills, /*hash=*/0, std::nullopt, cursor);

  EXPECT_EQ(d.reason, SelectionReason::NO_PEERS_AVAILABLE);
  EXPECT_FALSE(d.server_id.has_value());
}

TEST(PrefillSelectorTest, NoPeersAvailableWhenAllRefusingTasks) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, false),
                                           prefill("B", true, false)};
  size_t cursor = 0;
  SelectionResult d = selectPrefill(prefills, 0, std::nullopt, cursor);
  EXPECT_EQ(d.reason, SelectionReason::NO_PEERS_AVAILABLE);
}

TEST(PrefillSelectorTest, EqualityMatchPicksStickyTarget) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, true, /*inFlight=*/5),
      prefill("B", true, true, /*inFlight=*/0)};
  size_t cursor = 0;
  // Sticky -> A even though B is less loaded.
  SelectionResult d =
      selectPrefill(prefills, /*hash=*/42, std::string{"A"}, cursor);
  EXPECT_EQ(d.reason, SelectionReason::EQUALITY_MATCH);
  ASSERT_TRUE(d.server_id.has_value());
  EXPECT_EQ(*d.server_id, "A");
}

TEST(PrefillSelectorTest, StickyFallsBackWhenTargetUnhealthy) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", /*healthy=*/false),
                                           prefill("B", true, true, 0)};
  size_t cursor = 0;
  SelectionResult d = selectPrefill(prefills, 42, std::string{"A"}, cursor);
  EXPECT_EQ(d.reason, SelectionReason::LEAST_INFLIGHT);
  ASSERT_TRUE(d.server_id.has_value());
  EXPECT_EQ(*d.server_id, "B");
}

TEST(PrefillSelectorTest, StickyFallsBackWhenTargetOverloaded) {
  std::vector<PrefillSnapshot> prefills = {
      prefill("A", true, true, /*inFlight=*/8, /*maxInFlight=*/8),
      prefill("B", true, true, /*inFlight=*/2, /*maxInFlight=*/8)};
  size_t cursor = 0;
  SelectionResult d = selectPrefill(prefills, 42, std::string{"A"}, cursor);
  EXPECT_EQ(d.reason, SelectionReason::LEAST_INFLIGHT);
  ASSERT_TRUE(d.server_id.has_value());
  EXPECT_EQ(*d.server_id, "B");
}

TEST(PrefillSelectorTest, LeastInflightWinsOverLoaded) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, true, 3),
                                           prefill("B", true, true, 1),
                                           prefill("C", true, true, 5)};
  size_t cursor = 0;
  SelectionResult d = selectPrefill(prefills, 0, std::nullopt, cursor);
  EXPECT_EQ(d.reason, SelectionReason::LEAST_INFLIGHT);
  ASSERT_TRUE(d.server_id.has_value());
  EXPECT_EQ(*d.server_id, "B");
}

TEST(PrefillSelectorTest, RoundRobinAmongTiedLeastLoaded) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, true, 0),
                                           prefill("B", true, true, 0),
                                           prefill("C", true, true, 0)};
  size_t cursor = 0;
  SelectionResult d1 = selectPrefill(prefills, 0, std::nullopt, cursor);
  SelectionResult d2 = selectPrefill(prefills, 0, std::nullopt, cursor);
  SelectionResult d3 = selectPrefill(prefills, 0, std::nullopt, cursor);
  SelectionResult d4 = selectPrefill(prefills, 0, std::nullopt, cursor);

  EXPECT_EQ(d1.reason, SelectionReason::ROUND_ROBIN);
  EXPECT_EQ(d2.reason, SelectionReason::ROUND_ROBIN);
  EXPECT_EQ(d3.reason, SelectionReason::ROUND_ROBIN);
  EXPECT_EQ(d4.reason, SelectionReason::ROUND_ROBIN);

  // Cursor advances each call -> A, B, C, then wraps to A.
  ASSERT_TRUE(d1.server_id && d2.server_id && d3.server_id && d4.server_id);
  EXPECT_EQ(*d1.server_id, "A");
  EXPECT_EQ(*d2.server_id, "B");
  EXPECT_EQ(*d3.server_id, "C");
  EXPECT_EQ(*d4.server_id, "A");
}

TEST(PrefillSelectorTest, HashOfZeroIgnoresSticky) {
  std::vector<PrefillSnapshot> prefills = {prefill("A", true, true, 5),
                                           prefill("B", true, true, 0)};
  size_t cursor = 0;
  SelectionResult d =
      selectPrefill(prefills, /*hash=*/0, std::string{"A"}, cursor);
  EXPECT_EQ(d.reason, SelectionReason::LEAST_INFLIGHT);
  ASSERT_TRUE(d.server_id.has_value());
  EXPECT_EQ(*d.server_id, "B");
}

TEST(PrefillSelectorTest, ReasonLabelMatchesEnum) {
  EXPECT_STREQ(reasonLabel(SelectionReason::EQUALITY_MATCH), "equality_match");
  EXPECT_STREQ(reasonLabel(SelectionReason::LEAST_INFLIGHT), "least_inflight");
  EXPECT_STREQ(reasonLabel(SelectionReason::ROUND_ROBIN), "round_robin");
  EXPECT_STREQ(reasonLabel(SelectionReason::NO_PEERS_AVAILABLE),
               "no_peers_available");
}

}  // namespace
}  // namespace tt::gateway
