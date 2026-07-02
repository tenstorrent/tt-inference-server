// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_prefill_handoff.hpp"

#include "dynamo/json_value_utils.hpp"
#include "sockets/socket_messages.hpp"

#include <utility>

namespace tt::dynamo {

DynamoPrefillHandoff dynamoPrefillHandoffFromPrefillResult(
    const tt::sockets::PrefillResultMessage& result,
    const std::string& selectedPrefillId) {
  DynamoPrefillHandoff handoff;
  handoff.selectedPrefillId = selectedPrefillId;
  handoff.error = result.error;
  handoff.generatedText = result.generatedText;
  handoff.tokenIds = result.tokenIds;
  handoff.remainingTokens = result.remainingTokens;
  handoff.migrationId = result.migrationId;
  handoff.kvPositionId =
      result.tokenIds.empty()
          ? std::optional<uint32_t>{}
          : std::optional<uint32_t>(
                static_cast<uint32_t>(result.tokenIds.size() - 1));
  handoff.decodeSlotId = result.slotId;
  handoff.temperature = result.temperature;
  handoff.topP = result.topP;
  handoff.topK = result.topK;
  handoff.fastMode = result.fastMode;
  handoff.cachedTokens = result.cachedTokens;
  handoff.tokenCount = static_cast<uint32_t>(result.tokenIds.size());
  handoff.routingReason = "dynamo_prefill";
  return handoff;
}

tt::sockets::PrefillResultMessage dynamoPrefillHandoffToPrefillResult(
    uint32_t taskId, const DynamoPrefillHandoff& handoff) {
  tt::sockets::PrefillResultMessage result(taskId);
  result.generatedText = handoff.generatedText;
  result.error = handoff.error;
  result.tokenIds = handoff.tokenIds;
  result.remainingTokens = handoff.remainingTokens;
  result.slotId = handoff.decodeSlotId;
  result.temperature = handoff.temperature;
  result.topP = handoff.topP;
  result.topK = handoff.topK;
  result.fastMode = handoff.fastMode;
  result.cachedTokens = handoff.cachedTokens;
  result.migrationId = handoff.migrationId.value_or(0);
  return result;
}

Json::Value dynamoPrefillHandoffToJson(const DynamoPrefillHandoff& handoff) {
  Json::Value out(Json::objectValue);
  out["error"] = handoff.error;
  out["generated_text"] = handoff.generatedText;
  Json::Value tokenIds(Json::arrayValue);
  for (int64_t tokenId : handoff.tokenIds) {
    tokenIds.append(Json::Value(static_cast<Json::Int64>(tokenId)));
  }
  out["token_ids"] = std::move(tokenIds);
  json_value::setOptional(out, "remaining_tokens", handoff.remainingTokens);
  json_value::setOptional(out, "migration_id", handoff.migrationId);
  json_value::setOptional(out, "kv_position_id", handoff.kvPositionId);
  json_value::setOptional(out, "decode_slot_id", handoff.decodeSlotId);
  if (handoff.temperature.has_value()) {
    out["temperature"] = *handoff.temperature;
  } else {
    out["temperature"] = Json::Value::null;
  }
  if (handoff.topP.has_value()) {
    out["top_p"] = *handoff.topP;
  } else {
    out["top_p"] = Json::Value::null;
  }
  json_value::setOptional(out, "top_k", handoff.topK);
  out["fast_mode"] = handoff.fastMode;
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
  handoff.error = json.get("error", false).asBool();
  handoff.generatedText = json.get("generated_text", "").asString();
  if (json.isMember("token_ids") && json["token_ids"].isArray()) {
    for (const auto& tokenId : json["token_ids"]) {
      handoff.tokenIds.push_back(tokenId.asInt64());
    }
  }
  handoff.remainingTokens = json_value::optionalInt(json, "remaining_tokens");
  handoff.migrationId = json_value::optionalUInt64(json, "migration_id");
  handoff.kvPositionId = json_value::optionalUInt32(json, "kv_position_id");
  handoff.decodeSlotId = json_value::optionalUInt32(json, "decode_slot_id");
  if (json.isMember("temperature") && !json["temperature"].isNull()) {
    handoff.temperature = json["temperature"].asFloat();
  }
  if (json.isMember("top_p") && !json["top_p"].isNull()) {
    handoff.topP = json["top_p"].asFloat();
  }
  handoff.topK = json_value::optionalInt(json, "top_k");
  if (json.isMember("fast_mode") && json["fast_mode"].isBool()) {
    handoff.fastMode = json["fast_mode"].asBool();
  }
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
  if (handoff.error) {
    return {true, ""};
  }
  if (!handoff.migrationId.has_value() || *handoff.migrationId == 0) {
    return {false, "Dynamo prefill handoff requires migration_id"};
  }
  if (!handoff.kvPositionId.has_value()) {
    return {false, "Dynamo prefill handoff requires kv_position_id"};
  }
  if (handoff.tokenIds.empty()) {
    return {false, "Dynamo prefill handoff requires token_ids"};
  }
  return {true, ""};
}

}  // namespace tt::dynamo
