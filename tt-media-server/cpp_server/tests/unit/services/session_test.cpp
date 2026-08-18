// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <gtest/gtest.h>

namespace {

// ---------------------------------------------------------------------------
// Session state machine — all valid and invalid transitions
// Valid transitions:
//   IDLE      -> PREPARED   (markPrepared)
//   IDLE      -> IN_FLIGHT  (markInFlight, fast path skipping PREPARED)
//   PREPARED  -> IN_FLIGHT  (markInFlight)
//   IN_FLIGHT -> IDLE       (clearInFlight)
// ---------------------------------------------------------------------------

// Test interface: the state transitions are protected (owned by SessionManager
// in production). A subclass re-exposes them so the state machine can be unit
// tested directly, without coupling the domain header to gtest test names.
struct TestableSession : tt::domain::Session {
  using tt::domain::Session::clearInFlight;
  using tt::domain::Session::markInFlight;
  using tt::domain::Session::markPrepared;
};

TEST(SessionState, InitialStateIsIdle) {
  TestableSession s;
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, MarkPreparedFromIdle) {
  TestableSession s;
  EXPECT_TRUE(s.markPrepared());
  EXPECT_TRUE(s.isPrepared());
  EXPECT_FALSE(s.isIdle());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, MarkPreparedFromPreparedReturnsFalseAndPreservesState) {
  TestableSession s;
  ASSERT_TRUE(s.markPrepared());
  EXPECT_FALSE(s.markPrepared());  // already PREPARED
  EXPECT_TRUE(s.isPrepared());
}

TEST(SessionState, MarkPreparedFromInFlightReturnsFalseAndPreservesState) {
  TestableSession s;
  ASSERT_TRUE(s.markPrepared());
  ASSERT_TRUE(s.markInFlight());
  EXPECT_FALSE(s.markPrepared());  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, MarkInFlightFromPrepared) {
  TestableSession s;
  ASSERT_TRUE(s.markPrepared());
  EXPECT_TRUE(s.markInFlight());
  EXPECT_TRUE(s.isInFlight());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isIdle());
}

TEST(SessionState, MarkInFlightFromIdle) {
  TestableSession s;
  EXPECT_TRUE(s.markInFlight());  // IDLE -> IN_FLIGHT is allowed (fast path)
  EXPECT_TRUE(s.isInFlight());
  EXPECT_FALSE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
}

TEST(SessionState, MarkInFlightFromInFlightReturnsFalseAndPreservesState) {
  TestableSession s;
  ASSERT_TRUE(s.markPrepared());
  ASSERT_TRUE(s.markInFlight());
  EXPECT_FALSE(s.markInFlight());  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromInFlightTransitionsToIdle) {
  TestableSession s;
  ASSERT_TRUE(s.markPrepared());
  ASSERT_TRUE(s.markInFlight());
  EXPECT_TRUE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromIdleReturnsFalse) {
  TestableSession s;
  EXPECT_FALSE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());  // state unchanged
}

TEST(SessionState, ClearInFlightFromPreparedReturnsFalseAndPreservesState) {
  TestableSession s;
  ASSERT_TRUE(s.markPrepared());
  EXPECT_FALSE(
      s.clearInFlight());  // PREPARED -> IDLE not allowed via clearInFlight
  EXPECT_TRUE(s.isPrepared());
}

}  // namespace
