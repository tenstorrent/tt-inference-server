// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/native_prefill_handoff.hpp"

#include <limits>
#include <utility>

namespace tt::dynamo {

namespace {

std::optional<uint64_t> jsonUInt64(const Json::Value& obj, const char* field) {
  if (!obj.isMember(field) || obj[field].isNull()) return std::nullopt;
  if (obj[field].isUInt64()) return obj[field].asUInt64();
  if (obj[field].isUInt()) return obj[field].asUInt();
  if (obj[field].isString()) {
    try {
      return static_cast<uint64_t>(std::stoull(obj[field].asString()));
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<uint32_t> jsonUInt32(const Json::Value& obj, const char* field) {
  auto value = jsonUInt64(obj, field);
  if (!value.has_value() ||
      *value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(*value);
}

void setOptional(Json::Value& obj, const char* field,
                 const std::optional<uint32_t>& value) {
  if (value.has_value()) {
    obj[field] = *value;
  } else {
    obj[field] = Json::Value::null;
  }
}

void setOptional(Json::Value& obj, const char* field,
                 const std::optional<uint64_t>& value) {
  if (value.has_value()) {
    obj[field] = static_cast<Json::UInt64>(*value);
  } else {
    obj[field] = Json::Value::null;
  }
}

}  // namespace

NativePrefillHandoff buildMetadataOnlyNativePrefillHandoff(
    uint64_t migrationId, uint32_t tokenCount,
    const std::string& selectedPrefillId) {
  NativePrefillHandoff handoff;
  handoff.migrationId = migrationId;
  handoff.kvPositionId = tokenCount > 0 ? tokenCount - 1 : 0;
  handoff.cachedTokens = 0;
  handoff.tokenCount = tokenCount;
  handoff.selectedPrefillId = selectedPrefillId;
  handoff.routingReason = "dynamo_native_prefill";
  return handoff;
}

Json::Value nativePrefillHandoffToJson(const NativePrefillHandoff& handoff) {
  Json::Value out(Json::objectValue);
  setOptional(out, "migration_id", handoff.migrationId);
  setOptional(out, "kv_position_id", handoff.kvPositionId);
  setOptional(out, "decode_slot_id", handoff.decodeSlotId);
  out["cached_tokens"] = handoff.cachedTokens;
  out["token_count"] = handoff.tokenCount;
  out["selected_prefill_id"] = handoff.selectedPrefillId;
  out["routing_reason"] = handoff.routingReason;
  return out;
}

Json::Value nativePrefillHandoffToEngineData(
    const NativePrefillHandoff& handoff) {
  Json::Value engineData(Json::objectValue);
  engineData["tt_prefill_handoff"] = nativePrefillHandoffToJson(handoff);
  return engineData;
}

Json::Value nativePrefillHandoffToDisaggregatedParams(
    const NativePrefillHandoff& handoff) {
  Json::Value kvTransferParams(Json::objectValue);
  kvTransferParams["tt_prefill_handoff"] = nativePrefillHandoffToJson(handoff);

  Json::Value disaggregatedParams(Json::objectValue);
  disaggregatedParams["kv_transfer_params"] = std::move(kvTransferParams);
  return disaggregatedParams;
}

const Json::Value* findNativePrefillHandoffJson(const Json::Value& raw) {
  if (raw.isMember("prefill_result") && raw["prefill_result"].isObject()) {
    const auto& prefillResult = raw["prefill_result"];
    if (prefillResult.isMember("disaggregated_params") &&
        prefillResult["disaggregated_params"].isObject()) {
      const auto& disaggregatedParams =
          prefillResult["disaggregated_params"];
      if (disaggregatedParams.isMember("kv_transfer_params") &&
          disaggregatedParams["kv_transfer_params"].isObject() &&
          disaggregatedParams["kv_transfer_params"].isMember(
              "tt_prefill_handoff") &&
          disaggregatedParams["kv_transfer_params"]["tt_prefill_handoff"]
              .isObject()) {
        return &disaggregatedParams["kv_transfer_params"]
                                    ["tt_prefill_handoff"];
      }
      if (disaggregatedParams.isMember("tt_prefill_handoff") &&
          disaggregatedParams["tt_prefill_handoff"].isObject()) {
        return &disaggregatedParams["tt_prefill_handoff"];
      }
    }
  }
  if (raw.isMember("disaggregated_params") &&
      raw["disaggregated_params"].isObject()) {
    const auto& disaggregatedParams = raw["disaggregated_params"];
    if (disaggregatedParams.isMember("kv_transfer_params") &&
        disaggregatedParams["kv_transfer_params"].isObject() &&
        disaggregatedParams["kv_transfer_params"].isMember(
            "tt_prefill_handoff") &&
        disaggregatedParams["kv_transfer_params"]["tt_prefill_handoff"]
            .isObject()) {
      return &disaggregatedParams["kv_transfer_params"]["tt_prefill_handoff"];
    }
  }
  if (raw.isMember("tt_prefill_handoff") &&
      raw["tt_prefill_handoff"].isObject()) {
    return &raw["tt_prefill_handoff"];
  }
  if (raw.isMember("engine_data") && raw["engine_data"].isObject() &&
      raw["engine_data"].isMember("tt_prefill_handoff") &&
      raw["engine_data"]["tt_prefill_handoff"].isObject()) {
    return &raw["engine_data"]["tt_prefill_handoff"];
  }
  if (raw.isMember("nvext") && raw["nvext"].isObject() &&
      raw["nvext"].isMember("engine_data") &&
      raw["nvext"]["engine_data"].isObject() &&
      raw["nvext"]["engine_data"].isMember("tt_prefill_handoff") &&
      raw["nvext"]["engine_data"]["tt_prefill_handoff"].isObject()) {
    return &raw["nvext"]["engine_data"]["tt_prefill_handoff"];
  }
  return nullptr;
}

NativePrefillHandoff parseNativePrefillHandoff(const Json::Value& json) {
  NativePrefillHandoff handoff;
  handoff.migrationId = jsonUInt64(json, "migration_id");
  handoff.kvPositionId = jsonUInt32(json, "kv_position_id");
  handoff.decodeSlotId = jsonUInt32(json, "decode_slot_id");
  if (json.isMember("cached_tokens") && json["cached_tokens"].isInt()) {
    handoff.cachedTokens = json["cached_tokens"].asInt();
  }
  handoff.tokenCount = jsonUInt32(json, "token_count").value_or(0);
  handoff.selectedPrefillId =
      json.get("selected_prefill_id", "").asString();
  handoff.routingReason = json.get("routing_reason", "").asString();

  return handoff;
}

NativePrefillHandoffValidation validateNativePrefillHandoffForDecode(
    const NativePrefillHandoff& handoff) {
  if (handoff.selectedPrefillId.empty()) {
    return {false, "Dynamo native prefill handoff requires selected_prefill_id"};
  }
  if (!handoff.migrationId.has_value()) {
    return {false, "Dynamo native prefill handoff requires migration_id"};
  }
  if (!handoff.kvPositionId.has_value()) {
    return {false, "Dynamo native prefill handoff requires kv_position_id"};
  }
  return {true, ""};
}

void applyNativePrefillHandoffToRequest(
    const NativePrefillHandoff& handoff, tt::domain::llm::LLMRequest& req) {
  req.dynamoNativePrefillHandoff = true;
  req.disaggregated = true;
  req.migrationId = *handoff.migrationId;
  req.kv_position_id = *handoff.kvPositionId;
  req.dynamoNativePrefillDecodeSlotId = handoff.decodeSlotId;
  req.dynamoNativePrefillCachedTokens = handoff.cachedTokens;
}

}  // namespace tt::dynamo
