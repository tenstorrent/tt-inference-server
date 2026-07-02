// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <optional>
#include <string>

#include "domain/llm/llm_request.hpp"

namespace tt::dynamo {

struct DynamoPrefillHandoffValidation {
  bool ok = false;
  std::string error;
};

struct DynamoPrefillHandoff {
  std::string selectedPrefillId;
  std::optional<uint64_t> migrationId;
  std::optional<uint32_t> kvPositionId;
  std::optional<uint32_t> decodeSlotId;
  int cachedTokens = 0;
  uint32_t tokenCount = 0;
  std::string routingReason = "dynamo";
};

DynamoPrefillHandoff buildMetadataOnlyDynamoPrefillHandoff(
    uint64_t migrationId, uint32_t tokenCount,
    const std::string& selectedPrefillId);

Json::Value dynamoPrefillHandoffToJson(const DynamoPrefillHandoff& handoff);
Json::Value dynamoPrefillHandoffToDisaggregatedParams(
    const DynamoPrefillHandoff& handoff);

const Json::Value* findDynamoPrefillHandoffJson(const Json::Value& raw);
DynamoPrefillHandoff parseDynamoPrefillHandoff(const Json::Value& json);

DynamoPrefillHandoffValidation validateDynamoPrefillHandoffForDecode(
    const DynamoPrefillHandoff& handoff);
void applyDynamoPrefillHandoffToRequest(
    const DynamoPrefillHandoff& handoff, tt::domain::llm::LLMRequest& req);

}  // namespace tt::dynamo
