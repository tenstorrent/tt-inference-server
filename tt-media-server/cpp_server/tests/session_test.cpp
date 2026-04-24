// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <gtest/gtest.h>

namespace {

// ---------------------------------------------------------------------------
// Session state machine — all valid and invalid transitions
// ---------------------------------------------------------------------------

TEST(SessionState, InitialStateIsIdle) {
  tt::domain::Session s;
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isInFlight());
}

TEST(SessionState, MarkInFlightFromIdle) {
  tt::domain::Session s;
  EXPECT_TRUE(s.markInFlight());
  EXPECT_TRUE(s.isInFlight());
  EXPECT_FALSE(s.isIdle());
}

TEST(SessionState, MarkInFlightFromNonIdleReturnsFalseAndPreservesState) {
  tt::domain::Session s;
  s.markInFlight();
  EXPECT_FALSE(s.markInFlight());  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromInFlightTransitionsToIdle) {
  tt::domain::Session s;
  s.markInFlight();
  EXPECT_TRUE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());
}

TEST(SessionState, ClearInFlightFromIdleReturnsFalse) {
  tt::domain::Session s;
  EXPECT_FALSE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());  // state unchanged
}

}  // namespace
