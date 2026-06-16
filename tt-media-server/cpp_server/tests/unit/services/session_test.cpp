// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <gtest/gtest.h>

// Friend class to access protected Session methods for state machine testing
class SessionTestHelper {
 public:
  static bool markPrepared(tt::domain::Session& s) { return s.markPrepared(); }
  static bool markInFlight(tt::domain::Session& s) { return s.markInFlight(); }
  static bool clearInFlight(tt::domain::Session& s) { return s.clearInFlight(); }
};

namespace {

// ---------------------------------------------------------------------------
// Session state machine — all valid and invalid transitions
// Valid transitions:
//   IDLE      -> PREPARED   (markPrepared)
//   IDLE      -> IN_FLIGHT  (markInFlight, fast path skipping PREPARED)
//   PREPARED  -> IN_FLIGHT  (markInFlight)
//   IN_FLIGHT -> IDLE       (clearInFlight)
// ---------------------------------------------------------------------------

TEST(SessionState, InitialStateIsIdle) {
  tt::domain::Session s(1u);
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, MarkPreparedFromIdle) {
  tt::domain::Session s(1u);
  EXPECT_TRUE(SessionTestHelper::markPrepared(s));
  EXPECT_TRUE(s.isPrepared());
  EXPECT_FALSE(s.isIdle());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, MarkPreparedFromPreparedReturnsFalseAndPreservesState) {
  tt::domain::Session s(1u);
  ASSERT_TRUE(SessionTestHelper::markPrepared(s));
  EXPECT_FALSE(SessionTestHelper::markPrepared(s));  // already PREPARED
  EXPECT_TRUE(s.isPrepared());
}

TEST(SessionState, MarkPreparedFromInFlightReturnsFalseAndPreservesState) {
  tt::domain::Session s(1u);
  ASSERT_TRUE(SessionTestHelper::markPrepared(s));
  ASSERT_TRUE(SessionTestHelper::markInFlight(s));
  EXPECT_FALSE(SessionTestHelper::markPrepared(s));  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, MarkInFlightFromPrepared) {
  tt::domain::Session s(1u);
  ASSERT_TRUE(SessionTestHelper::markPrepared(s));
  EXPECT_TRUE(SessionTestHelper::markInFlight(s));
  EXPECT_TRUE(s.isInFlight());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isIdle());
}

TEST(SessionState, MarkInFlightFromIdle) {
  tt::domain::Session s(1u);
  EXPECT_TRUE(SessionTestHelper::markInFlight(s));  // IDLE -> IN_FLIGHT is allowed (fast path)
  EXPECT_TRUE(s.isInFlight());
  EXPECT_FALSE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
}

TEST(SessionState, MarkInFlightFromInFlightReturnsFalseAndPreservesState) {
  tt::domain::Session s(1u);
  ASSERT_TRUE(SessionTestHelper::markPrepared(s));
  ASSERT_TRUE(SessionTestHelper::markInFlight(s));
  EXPECT_FALSE(SessionTestHelper::markInFlight(s));  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromInFlightTransitionsToIdle) {
  tt::domain::Session s(1u);
  ASSERT_TRUE(SessionTestHelper::markPrepared(s));
  ASSERT_TRUE(SessionTestHelper::markInFlight(s));
  EXPECT_TRUE(SessionTestHelper::clearInFlight(s));
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isPrepared());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromIdleReturnsFalse) {
  tt::domain::Session s(1u);
  EXPECT_FALSE(SessionTestHelper::clearInFlight(s));
  EXPECT_TRUE(s.isIdle());  // state unchanged
}

TEST(SessionState, ClearInFlightFromPreparedReturnsFalseAndPreservesState) {
  tt::domain::Session s(1u);
  ASSERT_TRUE(SessionTestHelper::markPrepared(s));
  EXPECT_FALSE(
      SessionTestHelper::clearInFlight(s));  // PREPARED -> IDLE not allowed via clearInFlight
  EXPECT_TRUE(s.isPrepared());
}

}  // namespace
