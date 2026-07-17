// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/prefill_result_mapping.hpp"

#include <utility>

#include "utils/id_generator.hpp"

namespace tt::dynamo {

namespace {

std::optional<uint32_t> optionalUInt(const Json::Value& value) {
  if (value.isNull()) return std::nullopt;
  if (value.isUInt()) return value.asUInt();
  if (value.isInt() && value.asInt() >= 0) {
    return static_cast<uint32_t>(value.asInt());
  }
  return std::nullopt;
}

std::optional<int> optionalInt(const Json::Value& value) {
  if (value.isNull()) return std::nullopt;
  if (value.isInt()) return value.asInt();
  if (value.isUInt()) return static_cast<int>(value.asUInt());
  return std::nullopt;
}

}  // namespace

Json::Value prefillResultToJson(
    const tt::sockets::PrefillResultMessage& message) {
  Json::Value out(Json::objectValue);
  out["task_id"] = message.taskId;
  out["generated_text"] = message.generatedText;
  out["error"] = message.error;
  Json::Value tokenIds(Json::arrayValue);
  for (uint32_t tokenId : message.tokenIds) {
    tokenIds.append(tokenId);
  }
  out["token_ids"] = std::move(tokenIds);
  if (message.remainingTokens.has_value()) {
    out["remaining_tokens"] = *message.remainingTokens;
  }
  if (message.slotId.has_value()) {
    out["slot_id"] = *message.slotId;
  }
  if (message.temperature.has_value()) {
    out["temperature"] = *message.temperature;
  }
  if (message.topP.has_value()) {
    out["top_p"] = *message.topP;
  }
  if (message.topK.has_value()) {
    out["top_k"] = *message.topK;
  }
  out["fast_mode"] = message.fastMode;
  out["cached_tokens"] = message.cachedTokens;
  out["migration_id"] = Json::UInt64(message.migrationId);
  return out;
}

std::optional<tt::sockets::PrefillResultMessage> prefillResultFromJson(
    const Json::Value& dynRaw) {
  const Json::Value* ttResult = nullptr;
  auto tryParams = [&ttResult](const Json::Value& params) {
    if (ttResult != nullptr || !params.isObject()) return;
    if (params.isMember("tt_prefill_result") &&
        params["tt_prefill_result"].isObject()) {
      ttResult = &params["tt_prefill_result"];
    }
  };

  if (dynRaw.isMember("prefill_result") &&
      dynRaw["prefill_result"].isObject()) {
    const auto& prefillResult = dynRaw["prefill_result"];
    tryParams(prefillResult["disaggregated_params"]);
    tryParams(prefillResult);
  }
  tryParams(dynRaw["disaggregated_params"]);
  if (dynRaw.isMember("extra_args") && dynRaw["extra_args"].isObject()) {
    const auto& extraArgs = dynRaw["extra_args"];
    tryParams(extraArgs["disaggregated_params"]);
    if (extraArgs.isMember("prefill_result") &&
        extraArgs["prefill_result"].isObject()) {
      tryParams(extraArgs["prefill_result"]["disaggregated_params"]);
      tryParams(extraArgs["prefill_result"]);
    }
  }

  if (ttResult == nullptr) return std::nullopt;

  auto message = std::optional<tt::sockets::PrefillResultMessage>(
      std::in_place,
      ttResult->get("task_id", tt::utils::TaskIDGenerator::generate())
          .asUInt());
  auto& result = *message;
  result.generatedText = ttResult->get("generated_text", "").asString();
  result.error = ttResult->get("error", false).asBool();
  if (ttResult->isMember("token_ids") && (*ttResult)["token_ids"].isArray()) {
    for (const auto& token : (*ttResult)["token_ids"]) {
      result.tokenIds.push_back(token.asUInt());
    }
  }
  result.remainingTokens = optionalInt((*ttResult)["remaining_tokens"]);
  result.slotId = optionalUInt((*ttResult)["slot_id"]);
  if (!(*ttResult)["temperature"].isNull()) {
    result.temperature = (*ttResult)["temperature"].asFloat();
  }
  if (!(*ttResult)["top_p"].isNull()) {
    result.topP = (*ttResult)["top_p"].asFloat();
  }
  result.topK = optionalInt((*ttResult)["top_k"]);
  result.fastMode = ttResult->get("fast_mode", false).asBool();
  result.cachedTokens = ttResult->get("cached_tokens", 0).asInt();
  if (ttResult->isMember("migration_id")) {
    result.migrationId = (*ttResult)["migration_id"].asUInt64();
  }
  return message;
}

}  // namespace tt::dynamo
