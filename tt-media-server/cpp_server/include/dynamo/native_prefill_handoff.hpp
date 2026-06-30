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
  int version = 1;
  std::string status = "metadata_only";
  std::string request_id;

  std::optional<uint64_t> migration_id;
  std::optional<uint32_t> kv_position_id;
  std::optional<uint32_t> decode_slot_id;
  int cached_tokens = 0;
  uint32_t token_count = 0;

  std::string selected_prefill_id;
  std::string prefill_instance_id;
  std::string routing_reason = "dynamo";
  std::string cancellation_token;
  std::optional<uint64_t> deadline_unix_ms;
  std::optional<uint32_t> timeout_ms;

  std::optional<uint32_t> capacity_max_inflight;
  std::optional<uint32_t> capacity_inflight;
  std::optional<bool> capacity_healthy;
  std::optional<bool> capacity_accepting_tasks;

  std::optional<uint32_t> cache_matched_blocks;
  std::optional<uint32_t> cache_matched_tokens;

  std::optional<uint64_t> mooncake_uuid;
  std::optional<uint32_t> mooncake_slot;
  uint32_t mooncake_layer_begin = 0;
  uint32_t mooncake_layer_end = 0;
  uint32_t mooncake_position_begin = 0;
  uint32_t mooncake_position_end = 0;
};

NativePrefillHandoff buildMetadataOnlyNativePrefillHandoff(
    const std::string& requestId, uint64_t migrationId, uint32_t tokenCount,
    const std::string& localPrefillId);

Json::Value nativePrefillHandoffToJson(const NativePrefillHandoff& handoff);
Json::Value nativePrefillHandoffToEngineData(
    const NativePrefillHandoff& handoff);
Json::Value nativePrefillHandoffToDisaggregatedParams(
    const NativePrefillHandoff& handoff);

const Json::Value* findNativePrefillHandoffJson(const Json::Value& raw);
NativePrefillHandoff parseNativePrefillHandoff(const Json::Value& json);

NativePrefillHandoffValidation validateNativePrefillHandoffForDecode(
    const NativePrefillHandoff& handoff);
void applyNativePrefillHandoffToRequest(
    const NativePrefillHandoff& handoff, tt::domain::llm::LLMRequest& req);

}  // namespace tt::dynamo
