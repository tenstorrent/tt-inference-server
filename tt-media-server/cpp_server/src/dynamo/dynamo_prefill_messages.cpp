// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_prefill_messages.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include "utils/conversation_hasher.hpp"
#include "utils/id_generator.hpp"

namespace tt::dynamo {

tt::sockets::PrefillRequestMessage dynamoGenerateRequestToPrefillRequest(
    const GenerateRequest& request) {
  if (request.raw.isMember("tt_prefill_request") &&
      request.raw["tt_prefill_request"].isObject()) {
    const Json::Value& raw = request.raw["tt_prefill_request"];
    auto message = tt::sockets::PrefillRequestMessage(
        raw.get("task_id", tt::utils::TaskIDGenerator::generate()).asUInt());
    if (raw["registration_hashes"].isArray()) {
      for (const auto& hash : raw["registration_hashes"]) {
        message.registrationHashes.push_back(hash.asUInt64());
      }
    }
    if (raw["token_ids"].isArray()) {
      for (const auto& token : raw["token_ids"]) {
        message.tokenIds.push_back(token.asUInt());
      }
    }
    if (raw.isMember("max_tokens") && !raw["max_tokens"].isNull()) {
      message.maxTokens = raw["max_tokens"].asInt();
    }
    if (raw.isMember("slot_id") && !raw["slot_id"].isNull()) {
      message.slotId = raw["slot_id"].asUInt();
    }
    if (raw.isMember("temperature") && !raw["temperature"].isNull()) {
      message.temperature = raw["temperature"].asFloat();
    }
    if (raw.isMember("top_p") && !raw["top_p"].isNull()) {
      message.topP = raw["top_p"].asFloat();
    }
    if (raw.isMember("top_k") && !raw["top_k"].isNull()) {
      message.topK = raw["top_k"].asInt();
    }
    message.fastMode = raw.get("fast_mode", false).asBool();
    message.decodePositionId = raw.get("decode_position_id", 0).asInt();
    message.decodeSkipTokens = raw.get("decode_skip_tokens", 0).asInt();
    return message;
  }

  auto message = tt::sockets::PrefillRequestMessage(
      tt::utils::TaskIDGenerator::generate());
  message.registrationHashes =
      tt::utils::computePrefixCachingInfoFromTokens(request.token_ids).hashes();
  message.tokenIds.assign(request.token_ids.begin(), request.token_ids.end());
  message.maxTokens = request.max_tokens;
  message.temperature = request.temperature;
  message.topP = request.top_p;
  message.topK = request.top_k;
  return message;
}

Json::Value prefillRequestToDynamoJson(
    const tt::sockets::PrefillRequestMessage& request) {
  Json::Value root(Json::objectValue);
  root["task_id"] = Json::Value(static_cast<Json::UInt>(request.taskId));

  Json::Value hashes(Json::arrayValue);
  for (uint64_t hash : request.registrationHashes) {
    hashes.append(Json::Value(static_cast<Json::UInt64>(hash)));
  }
  root["registration_hashes"] = std::move(hashes);

  Json::Value tokens(Json::arrayValue);
  for (uint32_t token : request.tokenIds) {
    tokens.append(Json::Value(static_cast<Json::UInt>(token)));
  }
  root["token_ids"] = std::move(tokens);

  root["max_tokens"] = request.maxTokens.has_value()
                           ? Json::Value(*request.maxTokens)
                           : Json::Value::null;
  root["slot_id"] = request.slotId.has_value()
                        ? Json::Value(static_cast<Json::UInt>(*request.slotId))
                        : Json::Value::null;
  root["temperature"] = request.temperature.has_value()
                            ? Json::Value(*request.temperature)
                            : Json::Value::null;
  root["top_p"] =
      request.topP.has_value() ? Json::Value(*request.topP) : Json::Value::null;
  root["top_k"] =
      request.topK.has_value() ? Json::Value(*request.topK) : Json::Value::null;
  root["fast_mode"] = request.fastMode;
  root["decode_position_id"] = request.decodePositionId;
  root["decode_skip_tokens"] = request.decodeSkipTokens;
  return root;
}

Json::Value buildDynamoPrefillGenerateBody(
    const tt::sockets::PrefillRequestMessage& request,
    const std::string& requestId) {
  Json::Value body(Json::objectValue);
  body["request_id"] = requestId;
  Json::Value tokens(Json::arrayValue);
  for (uint32_t token : request.tokenIds) {
    tokens.append(Json::Value(static_cast<Json::UInt>(token)));
  }
  body["token_ids"] = std::move(tokens);

  Json::Value stop(Json::objectValue);
  stop["max_tokens"] = request.maxTokens.value_or(1);
  body["stop_conditions"] = std::move(stop);

  Json::Value sampling(Json::objectValue);
  if (request.temperature.has_value()) {
    sampling["temperature"] = *request.temperature;
  }
  if (request.topP.has_value()) sampling["top_p"] = *request.topP;
  if (request.topK.has_value()) sampling["top_k"] = *request.topK;
  body["sampling_options"] = std::move(sampling);

  body["tt_prefill_request"] = prefillRequestToDynamoJson(request);
  return body;
}

}  // namespace tt::dynamo
