// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_prefill_handoff.hpp"

#include "dynamo/json_value_utils.hpp"

namespace tt::dynamo {

DynamoPrefillHandoff buildMetadataOnlyDynamoPrefillHandoff(
    uint64_t migrationId, uint32_t tokenCount,
    const std::string& selectedPrefillId) {
  DynamoPrefillHandoff handoff;
  handoff.migrationId = migrationId;
  handoff.kvPositionId = tokenCount > 0 ? tokenCount - 1 : 0;
  handoff.cachedTokens = 0;
  handoff.tokenCount = tokenCount;
  handoff.selectedPrefillId = selectedPrefillId;
  handoff.routingReason = "dynamo_prefill";
  return handoff;
}

Json::Value dynamoPrefillHandoffToJson(const DynamoPrefillHandoff& handoff) {
  Json::Value out(Json::objectValue);
  json_value::setOptional(out, "migration_id", handoff.migrationId);
  json_value::setOptional(out, "kv_position_id", handoff.kvPositionId);
  json_value::setOptional(out, "decode_slot_id", handoff.decodeSlotId);
  out["cached_tokens"] = handoff.cachedTokens;
  out["token_count"] = handoff.tokenCount;
  out["selected_prefill_id"] = handoff.selectedPrefillId;
  out["routing_reason"] = handoff.routingReason;
  return out;
}

Json::Value dynamoPrefillHandoffToDisaggregatedParams(
    const DynamoPrefillHandoff& handoff) {
  Json::Value disaggregatedParams(Json::objectValue);
  disaggregatedParams["tt_prefill_handoff"] =
      dynamoPrefillHandoffToJson(handoff);
  return disaggregatedParams;
}

const Json::Value* findDynamoPrefillHandoffJson(const Json::Value& raw) {
  if (raw.isMember("prefill_result") && raw["prefill_result"].isObject()) {
    const auto& prefillResult = raw["prefill_result"];
    if (prefillResult.isMember("disaggregated_params") &&
        prefillResult["disaggregated_params"].isObject()) {
      const auto& disaggregatedParams =
          prefillResult["disaggregated_params"];
      if (disaggregatedParams.isMember("tt_prefill_handoff") &&
          disaggregatedParams["tt_prefill_handoff"].isObject()) {
        return &disaggregatedParams["tt_prefill_handoff"];
      }
    }
  }
  if (raw.isMember("disaggregated_params") &&
      raw["disaggregated_params"].isObject()) {
    const auto& disaggregatedParams = raw["disaggregated_params"];
    if (disaggregatedParams.isMember("tt_prefill_handoff") &&
        disaggregatedParams["tt_prefill_handoff"].isObject()) {
      return &disaggregatedParams["tt_prefill_handoff"];
    }
  }
  return nullptr;
}

DynamoPrefillHandoff parseDynamoPrefillHandoff(const Json::Value& json) {
  DynamoPrefillHandoff handoff;
  handoff.migrationId = json_value::optionalUInt64(json, "migration_id");
  handoff.kvPositionId = json_value::optionalUInt32(json, "kv_position_id");
  handoff.decodeSlotId = json_value::optionalUInt32(json, "decode_slot_id");
  if (json.isMember("cached_tokens") && json["cached_tokens"].isInt()) {
    handoff.cachedTokens = json["cached_tokens"].asInt();
  }
  handoff.tokenCount = json_value::optionalUInt32(json, "token_count")
                           .value_or(0);
  handoff.selectedPrefillId =
      json.get("selected_prefill_id", "").asString();
  handoff.routingReason = json.get("routing_reason", "").asString();

  return handoff;
}

DynamoPrefillHandoffValidation validateDynamoPrefillHandoffForDecode(
    const DynamoPrefillHandoff& handoff) {
  if (handoff.selectedPrefillId.empty()) {
    return {false, "Dynamo prefill handoff requires selected_prefill_id"};
  }
  if (!handoff.migrationId.has_value()) {
    return {false, "Dynamo prefill handoff requires migration_id"};
  }
  if (!handoff.kvPositionId.has_value()) {
    return {false, "Dynamo prefill handoff requires kv_position_id"};
  }
  return {true, ""};
}

void applyDynamoPrefillHandoffToRequest(
    const DynamoPrefillHandoff& handoff, tt::domain::llm::LLMRequest& req) {
  req.dynamoNativePrefillHandoff = true;
  req.disaggregated = true;
  req.migrationId = *handoff.migrationId;
  req.kv_position_id = *handoff.kvPositionId;
  req.dynamoNativePrefillDecodeSlotId = handoff.decodeSlotId;
  req.dynamoNativePrefillCachedTokens = handoff.cachedTokens;
}

}  // namespace tt::dynamo
