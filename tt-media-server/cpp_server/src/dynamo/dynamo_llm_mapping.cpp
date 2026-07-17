// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_llm_mapping.hpp"

#include <optional>
#include <vector>

#include "config/settings.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/id_generator.hpp"

namespace tt::dynamo {

namespace {

std::vector<uint64_t> computeRegistrationHashes(
    const std::vector<uint32_t>& tokenIds) {
  return tt::utils::computePrefixCachingInfoFromTokens(tokenIds).hashes();
}

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

std::shared_ptr<tt::domain::llm::LLMRequest> buildLLMRequestFromGenerateRequest(
    const GenerateRequest& dyn) {
  auto req = std::make_shared<tt::domain::llm::LLMRequest>(
      tt::utils::TaskIDGenerator::generate());
  req->stream = true;
  req->skip_apply_chat_template = true;
  req->skip_text_decode = true;
  req->prompt = dyn.token_ids;
  req->prompt_tokens_count = static_cast<int>(dyn.token_ids.size());
  req->full_prompt_tokens_count = req->prompt_tokens_count;

  if (!dyn.model.empty()) req->model = dyn.model;
  req->max_tokens =
      dyn.max_tokens.value_or(static_cast<int>(tt::config::maxContextLength()));
  if (dyn.min_tokens.has_value()) req->min_tokens = *dyn.min_tokens;
  req->stop_token_ids = dyn.stop_token_ids;
  req->stop = dyn.stop;
  req->ignore_eos = dyn.ignore_eos;

  if (dyn.temperature.has_value()) req->temperature = *dyn.temperature;
  if (dyn.top_p.has_value()) req->top_p = *dyn.top_p;
  if (dyn.top_k.has_value()) req->top_k = *dyn.top_k;
  if (dyn.seed.has_value()) req->seed = *dyn.seed;
  if (dyn.frequency_penalty.has_value())
    req->frequency_penalty = *dyn.frequency_penalty;
  if (dyn.presence_penalty.has_value())
    req->presence_penalty = *dyn.presence_penalty;
  if (dyn.repetition_penalty.has_value())
    req->repetition_penalty = *dyn.repetition_penalty;

  const std::string prevResponseId =
      dyn.raw.get("previous_response_id", "").asString();
  if (!prevResponseId.empty()) req->previousResponseId = prevResponseId;

  std::string currentId = dyn.raw.get("id", "").asString();
  if (currentId.empty()) currentId = dyn.raw.get("request_id", "").asString();
  if (!currentId.empty()) req->responseId = currentId;

  return req;
}

tt::sockets::PrefillRequestMessage buildPrefillRequestMessage(
    const GenerateRequest& dyn) {
  auto message = tt::sockets::PrefillRequestMessage(
      tt::utils::TaskIDGenerator::generate());
  message.registrationHashes = computeRegistrationHashes(dyn.token_ids);
  message.tokenIds = dyn.token_ids;
  message.maxTokens = dyn.max_tokens;
  message.temperature = dyn.temperature;
  message.topP = dyn.top_p;
  message.topK = dyn.top_k;

  const Json::Value& hints =
      dyn.raw.isMember("extra_args") ? dyn.raw["extra_args"] : dyn.raw;
  if (hints.isObject() && hints.isMember("tt_prefill_request") &&
      hints["tt_prefill_request"].isObject()) {
    const auto& ttReq = hints["tt_prefill_request"];
    message.slotId = optionalUInt(ttReq["slot_id"]);
    if (auto maxTokens = optionalInt(ttReq["max_tokens"])) {
      message.maxTokens = *maxTokens;
    }
    message.decodePositionId = ttReq.get("decode_position_id", 0).asInt();
    message.decodeSkipTokens = ttReq.get("decode_skip_tokens", 0).asInt();
    message.fastMode = ttReq.get("fast_mode", false).asBool();
  }
  return message;
}

TokenChunk tokenChunkFromStreamChunk(
    const tt::domain::llm::LLMStreamChunk& chunk, bool isFinal) {
  TokenChunk out;
  if (!chunk.choices.empty() && chunk.choices.front().token_id.has_value()) {
    out.token_ids = {static_cast<uint32_t>(*chunk.choices.front().token_id)};
  }
  if (isFinal) {
    if (!chunk.choices.empty()) {
      out.finish_reason = chunk.choices.front().finish_reason.value_or("stop");
    } else {
      out.finish_reason = "stop";
    }
  }
  return out;
}

}  // namespace tt::dynamo
