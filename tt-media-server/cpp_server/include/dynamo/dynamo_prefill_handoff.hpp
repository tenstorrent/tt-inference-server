// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tt::sockets {
struct PrefillResultMessage;
}

namespace tt::dynamo {

struct DynamoPrefillHandoffValidation {
  bool ok = false;
  std::string error;
};

struct DynamoPrefillHandoff {
  std::string selectedPrefillId;
  bool error = false;
  std::string generatedText;
  std::vector<uint32_t> tokenIds;
  std::optional<int> remainingTokens;
  std::optional<uint64_t> migrationId;
  std::optional<uint32_t> kvPositionId;
  std::optional<uint32_t> decodeSlotId;
  std::optional<float> temperature;
  std::optional<float> topP;
  std::optional<int> topK;
  bool fastMode = false;
  int cachedTokens = 0;
  uint32_t tokenCount = 0;
  std::string routingReason = "dynamo";
};

DynamoPrefillHandoff dynamoPrefillHandoffFromPrefillResult(
    const tt::sockets::PrefillResultMessage& result,
    const std::string& selectedPrefillId);
tt::sockets::PrefillResultMessage dynamoPrefillHandoffToPrefillResult(
    uint32_t taskId, const DynamoPrefillHandoff& handoff);

Json::Value dynamoPrefillHandoffToJson(const DynamoPrefillHandoff& handoff);
Json::Value dynamoPrefillHandoffToDisaggregatedParams(
    const DynamoPrefillHandoff& handoff);

const Json::Value* findDynamoPrefillHandoffJson(const Json::Value& raw);
DynamoPrefillHandoff parseDynamoPrefillHandoff(const Json::Value& json);

DynamoPrefillHandoffValidation validateDynamoPrefillHandoffForDecode(
    const DynamoPrefillHandoff& handoff);

}  // namespace tt::dynamo
