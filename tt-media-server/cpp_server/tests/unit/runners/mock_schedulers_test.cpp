// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "runtime/runners/blaze_runner/mock_scheduler.hpp"

namespace tt::runners::blaze {
namespace {

namespace sch = tt_llm_engine::scheduler;

sch::ISRequest makeAllocate(uint32_t requestId) {
  return {.type = sch::RequestType::ALLOCATE,
          .request_id = requestId,
          .tokens = {},
          .gen = {}};
}

sch::ISRequest makeSubmit(uint32_t slotId, uint32_t maxNewTokens,
                          std::vector<uint32_t> tokens = {1, 2, 3}) {
  sch::ISRequest req{};
  req.type = sch::RequestType::SUBMIT;
  req.slot_id = slotId;
  req.tokens = std::move(tokens);
  req.gen.max_new_tokens = maxNewTokens;
  return req;
}

// Block (with a short poll loop) until an output is available or `timeout`
// elapses. The async emitter thread produces tokens on its own schedule, so
// callers can no longer assume try_pop_output succeeds the instant
// push_request returns.
bool waitForOutput(
    MockDecodeScheduler& scheduler, sch::OutputMessage& out,
    std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (scheduler.try_pop_output(out)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  } while (std::chrono::steady_clock::now() < deadline);
  return scheduler.try_pop_output(out);
}

// Allocates a slot, submits a SUBMIT with `promptLen` prompt tokens requesting
// `maxNewTokens`, and returns the wall-clock span of the decode phase itself:
// from the first emitted token to the last. Timing deliberately starts at the
// first token rather than at push_request because the pipeline-fill time-to-
// first-token now scales with prompt length; decode throughput - the thing
// under test - is what must stay input-independent. `tokenCount` receives how
// many outputs were produced.
std::chrono::microseconds timeDecode(uint32_t maxNewTokens, size_t promptLen,
                                     uint32_t& tokenCount,
                                     MockDecodeSchedulerConfig cfg) {
  MockDecodeScheduler scheduler(4, std::move(cfg));
  scheduler.start();
  EXPECT_TRUE(scheduler.push_request(makeAllocate(1)));
  sch::SchedulerResponse alloc{};
  EXPECT_TRUE(scheduler.try_pop_response(alloc));

  const std::vector<uint32_t> prompt(promptLen, 7u);
  EXPECT_TRUE(
      scheduler.push_request(makeSubmit(alloc.slot_id, maxNewTokens, prompt)));

  sch::OutputMessage out{};
  std::chrono::steady_clock::time_point firstAt{};
  std::chrono::steady_clock::time_point lastAt{};
  tokenCount = 0;
  while (tokenCount < maxNewTokens && waitForOutput(scheduler, out)) {
    const auto now = std::chrono::steady_clock::now();
    if (tokenCount == 0) {
      firstAt = now;
    }
    lastAt = now;
    ++tokenCount;
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(lastAt -
                                                               firstAt);
}

TEST(MockSchedulerTest, PrefillSubmitCompletesWithNoDecodeTokens) {
  MockPrefillScheduler scheduler(4);
  scheduler.start();

  ASSERT_TRUE(scheduler.push_request(makeAllocate(1)));
  sch::SchedulerResponse allocateResponse{};
  ASSERT_TRUE(scheduler.try_pop_response(allocateResponse));
  const uint32_t slotId = allocateResponse.slot_id;

  ASSERT_TRUE(scheduler.push_request(makeSubmit(slotId, 0)));

  sch::OutputMessage output{};
  ASSERT_TRUE(scheduler.try_pop_output(output));
  EXPECT_TRUE(output.prefill_complete);
  EXPECT_EQ(output.real_pos, 3u);
  EXPECT_EQ(output.token_id, sch::EMPTY_TOKEN);
  EXPECT_FALSE(scheduler.try_pop_output(output));
}

TEST(MockSchedulerTest, DecodeSubmitEmitsMaxNewTokens) {
  // Keep this test fast: default-constructed config is already zero-latency,
  // which is exactly what we want here (no prefill stall, no decode spacing).
  MockDecodeScheduler scheduler(4, MockDecodeSchedulerConfig{});
  scheduler.start();

  ASSERT_TRUE(scheduler.push_request(makeAllocate(1)));
  sch::SchedulerResponse allocateResponse{};
  ASSERT_TRUE(scheduler.try_pop_response(allocateResponse));
  const uint32_t slotId = allocateResponse.slot_id;

  ASSERT_TRUE(scheduler.push_request(makeSubmit(slotId, 3)));

  for (uint32_t i = 0; i < 3; ++i) {
    sch::OutputMessage output{};
    ASSERT_TRUE(waitForOutput(scheduler, output)) << "missing token " << i;
    EXPECT_EQ(output.tokens_generated, i + 1);
    EXPECT_EQ(output.is_complete, i + 1 == 3);
  }
}

TEST(MockSchedulerTest, DecodeSubmitWithZeroMaxNewTokensCompletesImmediately) {
  MockDecodeScheduler scheduler(4, MockDecodeSchedulerConfig{});
  scheduler.start();

  ASSERT_TRUE(scheduler.push_request(makeAllocate(1)));
  sch::SchedulerResponse allocateResponse{};
  ASSERT_TRUE(scheduler.try_pop_response(allocateResponse));
  const uint32_t slotId = allocateResponse.slot_id;

  ASSERT_TRUE(scheduler.push_request(makeSubmit(slotId, /*maxNewTokens=*/0)));

  sch::OutputMessage output{};
  ASSERT_TRUE(waitForOutput(scheduler, output));
  EXPECT_TRUE(output.is_complete);
  EXPECT_EQ(output.tokens_generated, 0u);
  EXPECT_EQ(output.token_id, sch::EMPTY_TOKEN);
  EXPECT_EQ(output.slot_id, slotId);

  // No further outputs - sleep a bit and confirm the emitter isn't still
  // producing phantom tokens.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(scheduler.try_pop_output(output));
}

// Regression: with the async emitter, EVICT/STOP used to drop the in-flight
// PendingJob but leave already-published tokens in core.outputs. The decode
// runner drains scheduler responses before outputs, so the EVICT ack would
// flip the slot to FREE before the stale token was processed, tripping the
// "unexpected token for IDLE slot" assert (or, if the slot was re-allocated
// first, silently misattributing the token to a new task).
TEST(MockSchedulerTest, EvictPurgesQueuedAndPendingOutputsForSlot) {
  MockDecodeScheduler scheduler(4, MockDecodeSchedulerConfig{});
  scheduler.start();

  // Two slots: we'll evict A and make sure B is unaffected.
  ASSERT_TRUE(scheduler.push_request(makeAllocate(1)));
  sch::SchedulerResponse allocA{};
  ASSERT_TRUE(scheduler.try_pop_response(allocA));
  ASSERT_TRUE(scheduler.push_request(makeAllocate(2)));
  sch::SchedulerResponse allocB{};
  ASSERT_TRUE(scheduler.try_pop_response(allocB));
  ASSERT_NE(allocA.slot_id, allocB.slot_id);

  // Submit a long decode on A so the emitter has plenty to publish, plus a
  // short one on B that should survive the EVICT untouched. Sleep briefly to
  // let the emitter thread run; with latency=0 it pumps everything in one
  // burst.
  constexpr uint32_t kLongA = 50;
  constexpr uint32_t kShortB = 3;
  ASSERT_TRUE(scheduler.push_request(makeSubmit(allocA.slot_id, kLongA)));
  ASSERT_TRUE(scheduler.push_request(makeSubmit(allocB.slot_id, kShortB)));
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // EVICT A. Per the contract, this must purge both the in-flight PendingJob
  // (if any tokens are left) AND any tokens already sitting in core.outputs.
  sch::ISRequest evict{};
  evict.type = sch::RequestType::EVICT;
  evict.slot_id = allocA.slot_id;
  ASSERT_TRUE(scheduler.push_request(evict));

  // The EVICT ack response must still arrive.
  sch::SchedulerResponse evictAck{};
  ASSERT_TRUE(scheduler.try_pop_response(evictAck));
  EXPECT_EQ(evictAck.slot_id, allocA.slot_id);
  EXPECT_EQ(evictAck.request_type, sch::RequestType::EVICT);

  // Drain all remaining outputs. None should belong to the evicted slot; the
  // surviving slot must still get exactly its kShortB tokens.
  uint32_t slotAOutputs = 0;
  uint32_t slotBOutputs = 0;
  sch::OutputMessage out{};
  while (waitForOutput(scheduler, out, std::chrono::milliseconds(50))) {
    if (out.slot_id == allocA.slot_id) {
      ++slotAOutputs;
    } else if (out.slot_id == allocB.slot_id) {
      ++slotBOutputs;
    }
  }
  EXPECT_EQ(slotAOutputs, 0u)
      << "stale token for evicted slot leaked through outputs queue";
  EXPECT_EQ(slotBOutputs, kShortB)
      << "EVICT on slot A spuriously dropped B's tokens";
}

TEST(MockSchedulerTest, PrefillRejectsContinue) {
  MockPrefillScheduler scheduler(4);
  scheduler.start();

  sch::ISRequest continueReq{};
  continueReq.type = sch::RequestType::CONTINUE;
  continueReq.slot_id = 0;
  ASSERT_TRUE(scheduler.push_request(continueReq));

  sch::SchedulerResponse response{};
  ASSERT_TRUE(scheduler.try_pop_response(response));
  EXPECT_NE(response.error_code, sch::request_error::kOk);
}

// Models the real decode engine's steady-state throughput. A single sequence
// retires one token per full pipeline traversal (autoregression: token N+1
// waits for token N), so its per-slot decode spacing is numStages *
// stageLatency (e.g. 64 * 44us = 2816us) - independent of prompt length. The
// familiar ~1 token / 44us (~22.7k tok/s) is the aggregate rate once ~64
// slots interleave in the pipeline, not a per-slot rate. The fill latency does
// scale with prompt length (more chunks to stream in), so this test measures
// the decode span from first token to last, isolating the per-slot spacing: it
// must depend only on decodeTokenLatency, never on the number of input (prompt)
// tokens. (A larger spacing than 2816us keeps the test fast while exercising
// the same invariant.)
TEST(MockSchedulerTest, DecodeThroughputIsFlatAndInputIndependent) {
  constexpr unsigned kTokenLatencyUs = 1000;
  constexpr uint32_t kMaxNewTokens = 50;
  const MockDecodeSchedulerConfig cfg{
      .decodeTokenLatency = std::chrono::microseconds(kTokenLatencyUs),
  };

  uint32_t shortCount = 0;
  uint32_t longCount = 0;
  const auto shortPrompt =
      timeDecode(kMaxNewTokens, /*promptLen=*/4, shortCount, cfg);
  const auto longPrompt =
      timeDecode(kMaxNewTokens, /*promptLen=*/8192, longCount, cfg);

  // Output count is governed solely by max_new_tokens; a 2048x larger prompt
  // yields exactly the same number of tokens.
  EXPECT_EQ(shortCount, kMaxNewTokens);
  EXPECT_EQ(longCount, kMaxNewTokens);

  // The measured span covers the K-1 gaps between K tokens. It is bounded
  // below by (K-1) * latency: the emitter's cv_.wait_until never returns early,
  // so tokens can't be spaced tighter than one latency apart. It is bounded
  // above by that floor plus an overshoot budget for OS scheduling jitter
  // (thread wake-up + consumer poll jitter). Both prompt sizes must land in
  // this same window - the window depends only on K and latency, which is
  // exactly the input-independence we want.
  const auto nominalUs = std::chrono::microseconds(
      static_cast<int64_t>(kMaxNewTokens - 1) * kTokenLatencyUs);
  const auto floorUs = nominalUs - nominalUs / 100;  // 1% clock-source grace
  const auto ceilUs = nominalUs + nominalUs / 2;     // +50% overshoot budget

  for (const auto elapsed : {shortPrompt, longPrompt}) {
    EXPECT_GE(elapsed, floorUs);
    EXPECT_LE(elapsed, ceilUs);
  }

  // Input-independence, stated explicitly: a 2048x larger prompt shifts the run
  // time by no more than the overshoot budget (run-to-run jitter only).
  // This makes the "prompt size doesn't change the time" intent explicit and
  // obvious to a future reader
  EXPECT_NEAR(static_cast<double>(shortPrompt.count()),
              static_cast<double>(longPrompt.count()),
              static_cast<double>((ceilUs - floorUs).count()));
}

}  // namespace
}  // namespace tt::runners::blaze
