// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <gtest/gtest.h>

namespace {

// ---------------------------------------------------------------------------
// Session state machine — all valid and invalid transitions
// Valid path: IDLE -> PREPARED -> IN_FLIGHT -> IDLE
// ---------------------------------------------------------------------------

TEST(SessionState, InitialStateIsIdle) {
  tt::domain::Session s;
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, MarkPreparedFromIdle) {
  tt::domain::Session s;
  EXPECT_TRUE(s.markPrepared());
  EXPECT_TRUE(s.isPrepared());
  EXPECT_FALSE(s.isIdle());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, MarkPreparedFromPreparedReturnsFalseAndPreservesState) {
  tt::domain::Session s;
  ASSERT_TRUE(s.markPrepared());
  EXPECT_FALSE(s.markPrepared());  // already PREPARED
  EXPECT_TRUE(s.isPrepared());
}

TEST(SessionState, MarkPreparedFromInFlightReturnsFalseAndPreservesState) {
  tt::domain::Session s;
  ASSERT_TRUE(s.markPrepared());
  ASSERT_TRUE(s.markInFlight());
  EXPECT_FALSE(s.markPrepared());  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, MarkInFlightFromPrepared) {
  tt::domain::Session s;
  ASSERT_TRUE(s.markPrepared());
  EXPECT_TRUE(s.markInFlight());
  EXPECT_TRUE(s.isInFlight());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isIdle());
}

TEST(SessionState, MarkInFlightFromIdleReturnsFalseAndPreservesState) {
  tt::domain::Session s;
  EXPECT_FALSE(s.markInFlight());  // IDLE -> IN_FLIGHT is not allowed
  EXPECT_TRUE(s.isIdle());
}

TEST(SessionState, MarkInFlightFromInFlightReturnsFalseAndPreservesState) {
  tt::domain::Session s;
  ASSERT_TRUE(s.markPrepared());
  ASSERT_TRUE(s.markInFlight());
  EXPECT_FALSE(s.markInFlight());  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromInFlightTransitionsToIdle) {
  tt::domain::Session s;
  ASSERT_TRUE(s.markPrepared());
  ASSERT_TRUE(s.markInFlight());
  EXPECT_TRUE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromIdleReturnsFalse) {
  tt::domain::Session s;
  EXPECT_FALSE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());  // state unchanged
}

TEST(SessionState, ClearInFlightFromPreparedReturnsFalseAndPreservesState) {
  tt::domain::Session s;
  ASSERT_TRUE(s.markPrepared());
  EXPECT_FALSE(
      s.clearInFlight());  // PREPARED -> IDLE not allowed via clearInFlight
  EXPECT_TRUE(s.isPrepared());
}

}  // namespace
