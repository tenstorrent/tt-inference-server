// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared multi-turn prefix-cache assertion body for the integration tests.
//
// Used by main_integration_test's reasoning multiturn test to drive a
// conversation and assert the prefix-cache matched prefix grows correctly. The
// body is model-agnostic so the same assertions cover both the plain and the
// think-marker-in-history scenarios (see the threshold note in the caller).
//
// Note on models: config::modelType() and the tokenizer are process-wide cached
// singletons (a `static const` and a `thread_local`), resolved once when the
// server stack first boots, so a single test binary cannot switch models. The
// in-process harness also has no Kimi/tiktoken encoder (cpp_server reuses the
// DeepSeek tokenizer for Kimi, which can't read tiktoken.model — in production
// the Dynamo frontend tokenizes Kimi). So Kimi's think-marker condition is
// reproduced under the DeepSeek tokenizer by feeding <think>…</think> tags in
// the assistant history rather than by running a Kimi-tokenized binary.

#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <vector>

#include "support/chat_request.hpp"
#include "support/http_client.hpp"
#include "support/test_server.hpp"
#include "support/worker_response.hpp"

namespace tt::test {

// Drive a multi-turn streaming conversation against a running TestServer and
// assert the prefix-cache matched prefix GROWS monotonically: every turn after
// the first must hit the cache (continuation) and advance the matched prefix by
// at least one full block as the conversation grows.
//
// This is model-agnostic on purpose. With a reasoning model whose chat template
// injects <think> markers into the prompt history (e.g. Kimi), the exact
// non-think matched-token count is not recoverable from the Sequence alone
// (kv_position_id == matched_tokens-1 + accumulated_think_tokens, and the think
// component is not separately exposed). So instead of asserting an exact count
// we assert the invariant the prefix-cache must uphold and that the multiturn
// bug violated: the matched prefix never resets or plateaus — it advances by a
// whole block every turn. The earlier bug registered corrupt blocks past the
// matched prefix, so the next turn matched no more than the prior turn did;
// that shows up here as a failure to advance by a block.
//
// Each user message must be long enough to add at least one full block, and the
// generated token (42) must differ from the first fed-back assistant token so
// the seed session and the next prompt diverge right after the prior prompt.
inline void verifyMultiTurnPrefixGrowth(
    TestServer& server, const std::vector<std::string>& userMessages,
    const std::string& assistantReply, uint32_t blockSize) {
  ASSERT_GE(userMessages.size(), 2u)
      << "need at least an opener plus one follow-up";

  ChatRequest convo;
  uint32_t prevKvPos = 0;
  bool havePrev = false;

  for (size_t turn = 0; turn < userMessages.size(); ++turn) {
    convo.user(userMessages[turn]).maxTokens(1).stream();
    const std::string body = convo.toJson();
    auto future = std::async(std::launch::async, [&server, body] {
      return sendAndReceive(server.host(), server.port(), body);
    });

    auto seq = server.taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "turn " << turn;

    if (turn == 0) {
      // First turn: fresh allocation, nothing to match.
      EXPECT_FALSE(seq->isContinuation()) << "turn 0 must allocate a session";
      EXPECT_FALSE(seq->getKVPositionId().has_value()) << "turn 0";
    } else {
      ASSERT_TRUE(seq->isContinuation())
          << "turn " << turn << " must hit the prefix cache (no reset/reject)";
      ASSERT_TRUE(seq->getKVPositionId().has_value()) << "turn " << turn;
      const uint32_t kvPos = *seq->getKVPositionId();

      if (!havePrev) {
        // First continuation: the opener formed at least one block, so the
        // matched prefix (kvPos+1 tokens) covers at least one block.
        EXPECT_GE(kvPos + 1, blockSize)
            << "turn " << turn
            << ": first continuation should match >= 1 block";
      } else {
        // Each later turn appended at least one block of conversation, so the
        // matched prefix must advance by at least one full block.
        EXPECT_GE(kvPos, prevKvPos + blockSize)
            << "turn " << turn << ": matched prefix must advance by >= one "
            << "block (prev kv_position_id=" << prevKvPos
            << ", this turn=" << kvPos << "). A plateau here is the multiturn "
            << "bug: corrupt blocks past the matched prefix.";
      }
      prevKvPos = kvPos;
      havePrev = true;
    }

    // Mock the worker: one token (42) + FINAL. 42 differs from the assistant
    // reply's first token so the seed/next-prompt divergence is at the prior
    // prompt boundary.
    WorkerResponse(seq->taskId)
        .token(42)
        .finalize()
        .sendTo(server.resultQueue());
    future.get();

    convo.assistant(assistantReply);  // history for the next turn
  }
}

}  // namespace tt::test
