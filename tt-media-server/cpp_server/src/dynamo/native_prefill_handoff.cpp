// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/native_prefill_handoff.hpp"

#include <limits>
#include <stdexcept>
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

std::optional<bool> jsonBool(const Json::Value& obj, const char* field) {
  if (!obj.isMember(field) || obj[field].isNull() || !obj[field].isBool()) {
    return std::nullopt;
  }
  return obj[field].asBool();
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

void setOptional(Json::Value& obj, const char* field,
                 const std::optional<bool>& value) {
  if (value.has_value()) {
    obj[field] = *value;
  } else {
    obj[field] = Json::Value::null;
  }
}

}  // namespace

NativePrefillHandoff buildMetadataOnlyNativePrefillHandoff(
    const std::string& requestId, uint64_t migrationId, uint32_t tokenCount,
    const std::string& localPrefillId) {
  NativePrefillHandoff handoff;
  handoff.request_id = requestId;
  handoff.migration_id = migrationId;
  handoff.kv_position_id = tokenCount > 0 ? tokenCount - 1 : 0;
  handoff.cached_tokens = 0;
  handoff.token_count = tokenCount;
  handoff.selected_prefill_id = localPrefillId;
  handoff.prefill_instance_id = localPrefillId;
  handoff.routing_reason = "dynamo_native_prefill";
  handoff.cancellation_token = requestId;
  handoff.mooncake_uuid = migrationId;
  handoff.mooncake_position_end = tokenCount;
  return handoff;
}

Json::Value nativePrefillHandoffToJson(const NativePrefillHandoff& handoff) {
  Json::Value out(Json::objectValue);
  out["version"] = handoff.version;
  out["status"] = handoff.status;
  out["request_id"] = handoff.request_id;
  setOptional(out, "migration_id", handoff.migration_id);
  setOptional(out, "kv_position_id", handoff.kv_position_id);
  setOptional(out, "decode_slot_id", handoff.decode_slot_id);
  out["cached_tokens"] = handoff.cached_tokens;
  out["token_count"] = handoff.token_count;
  out["selected_prefill_id"] = handoff.selected_prefill_id;
  out["prefill_instance_id"] = handoff.prefill_instance_id;
  out["routing_reason"] = handoff.routing_reason;
  out["cancellation_token"] = handoff.cancellation_token;
  setOptional(out, "deadline_unix_ms", handoff.deadline_unix_ms);
  setOptional(out, "timeout_ms", handoff.timeout_ms);

  Json::Value capacity(Json::objectValue);
  setOptional(capacity, "max_inflight", handoff.capacity_max_inflight);
  setOptional(capacity, "inflight", handoff.capacity_inflight);
  setOptional(capacity, "healthy", handoff.capacity_healthy);
  setOptional(capacity, "accepting_tasks", handoff.capacity_accepting_tasks);
  out["capacity_snapshot"] = std::move(capacity);

  Json::Value cache(Json::objectValue);
  setOptional(cache, "matched_blocks", handoff.cache_matched_blocks);
  setOptional(cache, "matched_tokens", handoff.cache_matched_tokens);
  out["cache_overlap"] = std::move(cache);

  Json::Value mooncake(Json::objectValue);
  setOptional(mooncake, "uuid", handoff.mooncake_uuid);
  setOptional(mooncake, "slot", handoff.mooncake_slot);
  mooncake["layer_begin"] = handoff.mooncake_layer_begin;
  mooncake["layer_end"] = handoff.mooncake_layer_end;
  mooncake["position_begin"] = handoff.mooncake_position_begin;
  mooncake["position_end"] = handoff.mooncake_position_end;
  mooncake["status"] = handoff.mooncake_status;
  mooncake["depends_on"] = handoff.mooncake_depends_on;
  out["mooncake_migration"] = std::move(mooncake);

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
  handoff.version = json.get("version", 1).asInt();
  handoff.status = json.get("status", "").asString();
  handoff.request_id = json.get("request_id", "").asString();
  handoff.migration_id = jsonUInt64(json, "migration_id");
  handoff.kv_position_id = jsonUInt32(json, "kv_position_id");
  handoff.decode_slot_id = jsonUInt32(json, "decode_slot_id");
  if (json.isMember("cached_tokens") && json["cached_tokens"].isInt()) {
    handoff.cached_tokens = json["cached_tokens"].asInt();
  }
  handoff.token_count = jsonUInt32(json, "token_count").value_or(0);
  handoff.selected_prefill_id = json.get("selected_prefill_id", "").asString();
  handoff.prefill_instance_id = json.get("prefill_instance_id", "").asString();
  handoff.routing_reason = json.get("routing_reason", "").asString();
  handoff.cancellation_token = json.get("cancellation_token", "").asString();
  handoff.deadline_unix_ms = jsonUInt64(json, "deadline_unix_ms");
  handoff.timeout_ms = jsonUInt32(json, "timeout_ms");

  if (json.isMember("capacity_snapshot") &&
      json["capacity_snapshot"].isObject()) {
    const auto& capacity = json["capacity_snapshot"];
    handoff.capacity_max_inflight = jsonUInt32(capacity, "max_inflight");
    handoff.capacity_inflight = jsonUInt32(capacity, "inflight");
    handoff.capacity_healthy = jsonBool(capacity, "healthy");
    handoff.capacity_accepting_tasks = jsonBool(capacity, "accepting_tasks");
  }

  if (json.isMember("cache_overlap") && json["cache_overlap"].isObject()) {
    const auto& cache = json["cache_overlap"];
    handoff.cache_matched_blocks = jsonUInt32(cache, "matched_blocks");
    handoff.cache_matched_tokens = jsonUInt32(cache, "matched_tokens");
  }

  if (json.isMember("mooncake_migration") &&
      json["mooncake_migration"].isObject()) {
    const auto& mooncake = json["mooncake_migration"];
    handoff.mooncake_uuid = jsonUInt64(mooncake, "uuid");
    handoff.mooncake_slot = jsonUInt32(mooncake, "slot");
    handoff.mooncake_layer_begin =
        jsonUInt32(mooncake, "layer_begin").value_or(0);
    handoff.mooncake_layer_end = jsonUInt32(mooncake, "layer_end").value_or(0);
    handoff.mooncake_position_begin =
        jsonUInt32(mooncake, "position_begin").value_or(0);
    handoff.mooncake_position_end =
        jsonUInt32(mooncake, "position_end").value_or(0);
    handoff.mooncake_status = mooncake.get("status", "").asString();
    handoff.mooncake_depends_on = mooncake.get("depends_on", "").asString();
  }

  return handoff;
}

NativePrefillHandoffValidation validateNativePrefillHandoffForDecode(
    const NativePrefillHandoff& handoff) {
  if (handoff.selected_prefill_id.empty()) {
    return {false, "Dynamo native prefill handoff requires selected_prefill_id"};
  }
  if (!handoff.migration_id.has_value()) {
    return {false, "Dynamo native prefill handoff requires migration_id"};
  }
  if (!handoff.kv_position_id.has_value()) {
    return {false, "Dynamo native prefill handoff requires kv_position_id"};
  }
  if (handoff.mooncake_status != "complete") {
    return {false,
            "Dynamo native prefill handoff is present, but Mooncake migration "
            "is not complete"};
  }
  if (handoff.mooncake_uuid.has_value() &&
      handoff.mooncake_uuid != handoff.migration_id) {
    return {false,
            "Dynamo native prefill handoff migration_id does not match "
            "mooncake_migration.uuid"};
  }
  return {true, ""};
}

void applyNativePrefillHandoffToRequest(
    const NativePrefillHandoff& handoff, tt::domain::llm::LLMRequest& req) {
  req.dynamo_native_prefill_handoff = true;
  req.disaggregated = true;
  req.migrationId = *handoff.migration_id;
  req.kv_position_id = *handoff.kv_position_id;
  req.dynamo_native_prefill_decode_slot_id = handoff.decode_slot_id;
  req.dynamo_native_prefill_cached_tokens = handoff.cached_tokens;
}

}  // namespace tt::dynamo
