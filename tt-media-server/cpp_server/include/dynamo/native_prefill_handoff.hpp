// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <optional>
#include <string>

#include "domain/llm/llm_request.hpp"

namespace tt::dynamo {

struct NativePrefillHandoffValidation {
  bool ok = false;
  std::string error;
};

struct NativePrefillHandoff {
  std::string selectedPrefillId;
  std::optional<uint64_t> migrationId;
  std::optional<uint32_t> kvPositionId;
  std::optional<uint32_t> decodeSlotId;
  int cachedTokens = 0;
  uint32_t tokenCount = 0;
  std::string routingReason = "dynamo";
};

NativePrefillHandoff buildMetadataOnlyNativePrefillHandoff(
    uint64_t migrationId, uint32_t tokenCount,
    const std::string& selectedPrefillId);

Json::Value nativePrefillHandoffToJson(const NativePrefillHandoff& handoff);
Json::Value nativePrefillHandoffToDisaggregatedParams(
    const NativePrefillHandoff& handoff);

const Json::Value* findNativePrefillHandoffJson(const Json::Value& raw);
NativePrefillHandoff parseNativePrefillHandoff(const Json::Value& json);

NativePrefillHandoffValidation validateNativePrefillHandoffForDecode(
    const NativePrefillHandoff& handoff);
void applyNativePrefillHandoffToRequest(
    const NativePrefillHandoff& handoff, tt::domain::llm::LLMRequest& req);

}  // namespace tt::dynamo
