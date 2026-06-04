// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/socket_messages.hpp"

#include <gtest/gtest.h>

#include "sockets/socket_serialization.hpp"

namespace tt::sockets {
namespace {

// These round-trip the prefill messages through the exact wire path used
// between decode and prefill (serializeMessage / deserializePayload). The
// write() and read() field lists are maintained by hand in two places, so the
// trace_id assertion here guards against a field being appended to one side
// only (which would silently corrupt every field after it).

TEST(SocketMessagesTest, PrefillRequestRoundTripPreservesTraceId) {
  PrefillRequestMessage msg{42};
  msg.registration_hashes = {1, 2, 3};
  msg.token_ids = {10, 20, 30};
  msg.max_tokens = 16;
  msg.slot_id = 7;
  msg.temperature = 0.7F;
  msg.top_p = 0.9F;
  msg.top_k = 50;
  msg.fast_mode = true;
  msg.number_of_decode_skip_tokens = 4;
  msg.trace_id = "decode-3a9f2b1c";

  auto bytes = wire::serializeMessage("prefill_request", msg);
  auto restored = wire::deserializePayload<PrefillRequestMessage>(bytes);

  EXPECT_EQ(restored.task_id, msg.task_id);
  EXPECT_EQ(restored.registration_hashes, msg.registration_hashes);
  EXPECT_EQ(restored.token_ids, msg.token_ids);
  EXPECT_EQ(restored.max_tokens, msg.max_tokens);
  EXPECT_EQ(restored.slot_id, msg.slot_id);
  ASSERT_TRUE(restored.temperature.has_value());
  EXPECT_FLOAT_EQ(*restored.temperature, *msg.temperature);
  EXPECT_EQ(restored.fast_mode, msg.fast_mode);
  EXPECT_EQ(restored.number_of_decode_skip_tokens,
            msg.number_of_decode_skip_tokens);
  EXPECT_EQ(restored.trace_id, msg.trace_id);
}

TEST(SocketMessagesTest, PrefillRequestRoundTripEmptyTraceId) {
  PrefillRequestMessage msg{1};
  msg.token_ids = {1, 2};

  auto bytes = wire::serializeMessage("prefill_request", msg);
  auto restored = wire::deserializePayload<PrefillRequestMessage>(bytes);

  EXPECT_EQ(restored.task_id, msg.task_id);
  EXPECT_EQ(restored.token_ids, msg.token_ids);
  EXPECT_TRUE(restored.trace_id.empty());
}

TEST(SocketMessagesTest, PrefillResultRoundTripPreservesTraceId) {
  PrefillResultMessage msg{99};
  msg.generated_text = "hello";
  msg.finished = true;
  msg.tokens_generated = 1;
  msg.processing_time_ms = 12.5;
  msg.token_ids = {5, 6, 7};
  msg.remaining_tokens = 3;
  msg.slot_id = 2;
  msg.temperature = 0.5F;
  msg.fast_mode = true;
  msg.trace_id = "prefill-deadbeef";

  auto bytes = wire::serializeMessage("prefill_result", msg);
  auto restored = wire::deserializePayload<PrefillResultMessage>(bytes);

  EXPECT_EQ(restored.task_id, msg.task_id);
  EXPECT_EQ(restored.generated_text, msg.generated_text);
  EXPECT_EQ(restored.finished, msg.finished);
  EXPECT_EQ(restored.tokens_generated, msg.tokens_generated);
  EXPECT_EQ(restored.token_ids, msg.token_ids);
  EXPECT_EQ(restored.remaining_tokens, msg.remaining_tokens);
  EXPECT_EQ(restored.slot_id, msg.slot_id);
  EXPECT_EQ(restored.fast_mode, msg.fast_mode);
  EXPECT_EQ(restored.trace_id, msg.trace_id);
}

}  // namespace
}  // namespace tt::sockets
