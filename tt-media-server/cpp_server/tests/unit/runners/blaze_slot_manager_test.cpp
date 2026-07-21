// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Regression test for the "1 ISL -> next request reads the previous request's
// KV" contamination bug.
//
// Background (see blaze_slot_manager.hpp / blaze_types.hpp):
//   A blaze slot moves FREE -> IDLE (allocate) -> RUNNING (submit) and, on
//   completion, RUNNING -> IDLE via SlotManager::setSlotAsIdle(). IDLE means
//   "allocated, no request running (slot retained for prefix cache)", so the
//   slot's KV write cursor `currentPosition` is intentionally kept - the idea
//   being a matching follow-up can resume from it.
//
//   `currentPosition` is only zeroed by SlotManager::clearSlotContext(), which
//   runs on the FULL eviction path (EVICT ack -> FREE). A short turn (e.g. 1
//   prompt token in, 1 token out) commits 0 full prefix-cache blocks, so it is
//   never registered in the prefix cache and never gets an EVICT. It therefore
//   sits in IDLE with a stale, non-zero `currentPosition`.
//
//   When that IDLE slot is handed to the NEXT request, if that request is a
//   fresh (non-continuation, prefix-miss) SUBMIT it does not override the KV
//   position (session_resolution only sets kv_position_id when matchedTokens >
//   0; blaze_utils::fillSequenceFields only sets req.position_id when the
//   sequence carries a KVPositionId). The scheduler then writes the new prompt
//   at the retained cursor and attends to the stale KV -> the new response is
//   contaminated by the previous request's tokens.
//
// This test pins the cpp_server-side invariant that a reusable IDLE slot must
// not present a stale KV cursor. It fails on the current code (cursor is
// retained across RUNNING -> IDLE). Reproducing the actual token bleed
// end-to-end would additionally need tt-llm-engine's real scheduler: the mock
// scheduler falls back to `tokens.size()` rather than the slot's retained
// cursor (mock_scheduler.hpp: basePosition = position_id.value_or(tokens
// .size())), so it masks the contamination.

#include "runtime/runners/blaze_runner/blaze_slot_manager.hpp"

#include <gtest/gtest.h>

#include <cstdint>

#include "runtime/runners/blaze_runner/blaze_types.hpp"

namespace tt::runners::blaze {
namespace {

constexpr uint32_t kMaxSlots = 4;
constexpr uint32_t kSlotId = 0;
constexpr uint32_t kTaskId = 4242;

// Drives one slot through a complete short turn: allocate -> submit ->
// (1 prompt token in, 1 token out) -> completion, leaving the slot back in
// IDLE with the KV cursor the scheduler last reported.
//
// The `currentPosition = 2` mirrors what BlazePrefillRunner/BlazeDecodeRunner
// write from OutputMessage (blaze_prefill_runner.cpp: currentPosition =
// output.real_pos; blaze_decode_runner.cpp: currentPosition =
// output.position_id) after a 1-in/1-out turn.
void runShortTurnLeavingSlotIdle(SlotManager& slotManager) {
  // FREE -> IDLE: memory ALLOCATE ack (handleAllocateAck).
  slotManager.setSlotState(kSlotId, SlotState::IDLE);

  // IDLE -> RUNNING: SUBMIT (handleTask), binds the task to the slot.
  slotManager.bindTaskToSlot(kTaskId, kSlotId);
  slotManager.setSlotState(kSlotId, SlotState::RUNNING);

  // Scheduler advances the KV write cursor as the turn runs.
  slotManager.getSlotContext(kSlotId).currentPosition = 2;

  // Completion drains the slot back to IDLE for reuse (setSlotAsIdle), the
  // path a normal finished request takes.
  slotManager.setSlotAsIdle(kSlotId);
}

// The bug: after a short turn the slot is IDLE (i.e. advertised as reusable via
// the legal IDLE -> RUNNING transition) yet still carries the previous turn's
// KV cursor. A fresh request routed here inherits that cursor and reads stale
// KV. A reusable slot must present a clean cursor.
TEST(BlazeSlotManagerTest, ReusableIdleSlotMustNotCarryStaleKvCursor) {
  SlotManager slotManager(kMaxSlots);

  runShortTurnLeavingSlotIdle(slotManager);

  const auto& slot = slotManager.getSlotContext(kSlotId);
  // Slot is reusable...
  ASSERT_EQ(slot.state, SlotState::IDLE);
  // ...but the KV cursor is still pointing past the previous request's tokens.
  EXPECT_EQ(slot.currentPosition, 0u)
      << "IDLE slot handed to the next request still points at KV position "
      << slot.currentPosition
      << "; a fresh SUBMIT would write on top of / attend to the previous "
         "request's KV and echo its tokens.";
}

// Control: the ONLY path that currently clears the cursor is a full eviction
// (clearSlotContext). The short-turn / prefix-miss case above never reaches it,
// which is why the stale cursor survives into slot reuse.
TEST(BlazeSlotManagerTest, EvictClearsKvCursor) {
  SlotManager slotManager(kMaxSlots);

  runShortTurnLeavingSlotIdle(slotManager);
  ASSERT_EQ(slotManager.getSlotContext(kSlotId).currentPosition, 2u)
      << "precondition: short turn leaves a stale cursor";

  // Full eviction (EVICT ack -> handleEvictAck -> clearSlotContext -> FREE).
  slotManager.clearSlotContext(kSlotId);

  const auto& slot = slotManager.getSlotContext(kSlotId);
  EXPECT_EQ(slot.state, SlotState::FREE);
  EXPECT_EQ(slot.currentPosition, 0u);
}

}  // namespace
}  // namespace tt::runners::blaze
