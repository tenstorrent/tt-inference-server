// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "domain/sentinel_values.hpp"
#include "sockets/socket_messages.hpp"
#include "sockets/socket_serialization.hpp"

namespace tt::sockets {
namespace {

template <typename T>
T roundTrip(std::string_view tag, const T& original) {
  const auto bytes = wire::serializeMessage(tag, original);
  EXPECT_EQ(wire::readMessageType(bytes), tag);
  return wire::deserializePayload<T>(bytes);
}

TEST(SocketMessagesTest, SlotReservationRequestRoundTrip) {
  SlotReservationRequestMessage original;
  original.taskId = 42;
  original.prefillServerId = "prefill-a";
  original.registrationHashes = {0xDEADBEEFULL, 0xCAFEBABEULL};
  original.hasPreviousResponseId = true;
  original.previousResponseId = "resp-prev-1";
  original.promptTokenCount = 2048;

  const auto restored =
      roundTrip<SlotReservationRequestMessage>(tags::SLOT_RESERVATION_REQUEST,
                                               original);

  EXPECT_EQ(restored.taskId, original.taskId);
  EXPECT_EQ(restored.prefillServerId, original.prefillServerId);
  EXPECT_EQ(restored.registrationHashes, original.registrationHashes);
  EXPECT_EQ(restored.hasPreviousResponseId, original.hasPreviousResponseId);
  EXPECT_EQ(restored.previousResponseId, original.previousResponseId);
  EXPECT_EQ(restored.promptTokenCount, original.promptTokenCount);
}

TEST(SocketMessagesTest, SlotReservationRequestRoundTripWithoutResponseId) {
  SlotReservationRequestMessage original;
  original.taskId = 7;
  original.prefillServerId = "prefill-b";
  original.registrationHashes = {1, 2, 3};
  original.promptTokenCount = 512;

  const auto restored =
      roundTrip<SlotReservationRequestMessage>(tags::SLOT_RESERVATION_REQUEST,
                                               original);

  EXPECT_FALSE(restored.hasPreviousResponseId);
  EXPECT_TRUE(restored.previousResponseId.empty());
}

TEST(SocketMessagesTest, SlotReservationResponseRoundTripSuccess) {
  SlotReservationResponseMessage original;
  original.taskId = 99;
  original.hasSlot = true;
  original.slotId = 3;
  original.decodePositionId = 128;
  original.decodeSkipTokens = 120;
  original.continuation = true;
  original.accumulatedThinkTokens = 8;
  original.error = false;

  const auto restored =
      roundTrip<SlotReservationResponseMessage>(tags::SLOT_RESERVATION_RESPONSE,
                                                original);

  EXPECT_EQ(restored.taskId, original.taskId);
  EXPECT_EQ(restored.hasSlot, original.hasSlot);
  EXPECT_EQ(restored.slotId, original.slotId);
  EXPECT_EQ(restored.decodePositionId, original.decodePositionId);
  EXPECT_EQ(restored.decodeSkipTokens, original.decodeSkipTokens);
  EXPECT_EQ(restored.continuation, original.continuation);
  EXPECT_EQ(restored.accumulatedThinkTokens, original.accumulatedThinkTokens);
  EXPECT_FALSE(restored.error);
  EXPECT_TRUE(restored.errorText.empty());
}

TEST(SocketMessagesTest, SlotReservationResponseRoundTripError) {
  SlotReservationResponseMessage original;
  original.taskId = 100;
  original.hasSlot = false;
  original.slotId = tt::domain::INVALID_SLOT_ID;
  original.error = true;
  original.errorText = "allocation_failed";

  const auto restored =
      roundTrip<SlotReservationResponseMessage>(tags::SLOT_RESERVATION_RESPONSE,
                                                original);

  EXPECT_TRUE(restored.error);
  EXPECT_EQ(restored.errorText, original.errorText);
  EXPECT_FALSE(restored.hasSlot);
}

}  // namespace
}  // namespace tt::sockets
