// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "api/error_response.hpp"
#include "api/llm_controller.hpp"
#include "domain/json_field.hpp"
#include "domain/llm/chat_completion_request.hpp"
#include "utils/id_generator.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::api {
namespace {
using namespace tt::domain::llm;

std::vector<uint32_t> parsePrompt(const Json::Value& prompt) {
  if (prompt.isString()) {
    return tt::utils::tokenizers::activeTokenizer().encode(prompt.asString());
  }
  if (!prompt.isArray() || prompt.empty()) {
    throw std::invalid_argument(
        "prompt must contain a non-empty token array or string");
  }
  std::vector<uint32_t> ids;
  ids.reserve(prompt.size());
  for (const auto& id : prompt) {
    if (!id.isUInt() || id.asUInt() > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument(
          "prompt token IDs must be non-negative integers");
    }
    if (tt::config::modelType() ==
            tt::config::ModelType::LLAMA_3_2_1B_INSTRUCT &&
        id.asUInt() >= 128256) {
      throw std::invalid_argument(
          "prompt token ID exceeds Llama-3.2-1B vocabulary");
    }
    ids.push_back(id.asUInt());
  }
  return ids;
}

std::vector<std::vector<uint32_t>> parsePrompts(const Json::Value& prompt) {
  std::vector<std::vector<uint32_t>> prompts;
  if (prompt.isArray() && !prompt.empty() &&
      (prompt[0].isArray() || prompt[0].isString())) {
    for (const auto& item : prompt) prompts.push_back(parsePrompt(item));
  } else {
    prompts.push_back(parsePrompt(prompt));
  }
  return prompts;
}
}  // namespace

struct LLMController::CompletionBatch {
  std::vector<std::shared_ptr<LLMRequest>> requests;
  size_t next = 0;
  bool returnTokenIds = false;
  Json::Value response;
  std::function<void(const drogon::HttpResponsePtr&)> callback;
  trantor::EventLoop* loop = nullptr;
};

void LLMController::completions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  const auto json = req->getJsonObject();
  if (!json || !json->isObject()) {
    callback(errorResponse(drogon::k400BadRequest, "Invalid JSON body",
                           "invalid_request_error"));
    return;
  }
  if (!service->isModelReady()) {
    callback(errorResponse(drogon::k503ServiceUnavailable, "Model is not ready",
                           "service_unavailable"));
    return;
  }
  auto batch = std::make_shared<CompletionBatch>();
  try {
    // Reject options this endpoint cannot honor, including seeds unsupported
    // by the existing engine, rather than silently ignoring them.
    for (const char* field : {"seed", "allowed_token_ids", "prompt_logprobs",
                              "truncate_prompt_tokens", "best_of", "logit_bias",
                              "guided_json", "guided_regex", "guided_choice",
                              "guided_grammar", "structured_outputs"}) {
      if (json->isMember(field) && !(*json)[field].isNull()) {
        throw std::invalid_argument(std::string(field) +
                                    " is not supported for completions");
      }
    }
    if (json->isMember("stop_token_ids")) {
      const auto& stopIds = (*json)["stop_token_ids"];
      if (!stopIds.isArray())
        throw std::invalid_argument("stop_token_ids must be an array");
      for (const auto& id : stopIds) {
        if (!id.isUInt() || id.asUInt() > std::numeric_limits<int32_t>::max() ||
            (tt::config::modelType() ==
                 tt::config::ModelType::LLAMA_3_2_1B_INSTRUCT &&
             id.asUInt() >= 128256)) {
          throw std::invalid_argument(
              "stop_token_ids contains an invalid token ID");
        }
      }
    }
    auto parameters = ChatCompletionRequest::fromJson(*json, 0);
    // The pinned engine's ignore_eos disables both EOS and explicit stop IDs.
    // Reject this combination instead of silently ignoring a requested stop.
    if (parameters.ignore_eos && !parameters.stop_token_ids.empty()) {
      throw std::invalid_argument(
          "ignore_eos=true with stop_token_ids is not supported by this "
          "engine");
    }
    if (parameters.stream || parameters.n != 1 || parameters.echo ||
        parameters.suffix) {
      throw std::invalid_argument(
          "completions supports stream=false, n=1, echo=false and no suffix; "
          "batch prompts for multiple choices");
    }
    if (!parameters.messages.empty())
      throw std::invalid_argument("use prompt for completions");
    if (parameters.model && !parameters.model->empty() &&
        *parameters.model != tt::config::toString(tt::config::model())) {
      throw std::invalid_argument(
          "requested model does not match the loaded model");
    }
    if (parameters.logprobs && *parameters.logprobs != 0) {
      throw std::invalid_argument(
          "alternative token logprobs are not supported");
    }
    if (parameters.presence_penalty != 0 || parameters.frequency_penalty != 0 ||
        parameters.use_beam_search || parameters.min_tokens != 0 ||
        !parameters.stop.empty() || parameters.min_p.value_or(0) != 0 ||
        parameters.repetition_penalty.value_or(1) != 1 ||
        parameters.length_penalty != 1) {
      throw std::invalid_argument(
          "requested sampling penalty or constraint is not supported");
    }
    const float temperature = parameters.temperature.value_or(1);
    const float topP = parameters.top_p.value_or(1);
    if (!std::isfinite(temperature) || temperature < 0 ||
        !std::isfinite(topP) || topP <= 0 || topP > 1) {
      throw std::invalid_argument(
          "temperature must be non-negative and top_p must be in (0, 1]");
    }
    const int topK = parameters.top_k.value_or(-1);
    if (topK != -1 && topK < 1)
      throw std::invalid_argument("top_k must be -1 or positive");
    if (tt::config::modelType() ==
            tt::config::ModelType::LLAMA_3_2_1B_INSTRUCT &&
        topK > 32) {
      throw std::invalid_argument("this Blaze model supports top_k up to 32");
    }
    const int maxTokens = parameters.max_tokens.value_or(16);
    if (maxTokens <= 0)
      throw std::invalid_argument("max_tokens must be positive");
    batch->returnTokenIds =
        json->isMember("return_token_ids") &&
        tt::domain::json_field::getBool((*json)["return_token_ids"],
                                        "return_token_ids");
    auto prompts = parsePrompts((*json)["prompt"]);
    if (prompts.size() > 1 && parameters.sessionId) {
      throw std::invalid_argument(
          "session_id is supported only for a single prompt");
    }
    for (auto& tokens : prompts) {
      if (tokens.empty() || tokens.size() + static_cast<size_t>(maxTokens) >
                                tt::config::maxContextLength()) {
        throw std::invalid_argument(
            "prompt is empty or prompt plus max_tokens exceeds the configured "
            "context length");
      }
      auto request =
          std::make_shared<LLMRequest>(tt::utils::TaskIDGenerator::generate());
      parameters.copyGenerationParams(*request);
      request->model = tt::config::toString(tt::config::model());
      request->max_tokens = maxTokens;
      request->prompt_tokens_count = static_cast<int>(tokens.size());
      request->full_prompt_tokens_count = request->prompt_tokens_count;
      request->prompt = std::move(tokens);
      // Keep every sampled token, including EOS and tokens with no decoded
      // text.
      request->skip_text_decode = true;
      batch->requests.push_back(std::move(request));
    }
  } catch (const std::exception& error) {
    callback(errorResponse(drogon::k400BadRequest, error.what(),
                           "invalid_request_error"));
    return;
  }
  batch->loop = trantor::EventLoop::getEventLoopOfCurrentThread();
  batch->callback = std::move(callback);
  batch->response["id"] =
      "cmpl-" + std::to_string(batch->requests.front()->task_id);
  batch->response["object"] = "text_completion";
  batch->response["model"] = *batch->requests.front()->model;
  batch->response["choices"] = Json::Value(Json::arrayValue);
  batch->response["usage"]["prompt_tokens"] = 0;
  batch->response["usage"]["completion_tokens"] = 0;
  batch->response["usage"]["total_tokens"] = 0;
  completeBatch(batch);
}

void LLMController::completeBatch(
    std::shared_ptr<CompletionBatch> batch) const {
  const auto request = batch->requests.at(batch->next);
  auto builder =
      [returnIds = batch->returnTokenIds,
       skipSpecial = request->skip_special_tokens,
       ignoreEos = request->ignore_eos,
       stopTokens =
           tt::utils::mapper::mapSamplingParams(*request).stop_token_ids](
          const LLMResponse& response) {
        Json::Value json;
        const auto& output = response.choices.at(0);
        json["created"] = Json::Int64(response.created);
        json["text"] = tt::utils::tokenizers::activeTokenizer().decode(
            output.token_ids, skipSpecial);
        auto finishReason = output.finish_reason.value_or("length");
        if (finishReason == "stop" || finishReason == "length") {
          const bool stopped =
              !ignoreEos && !output.token_ids.empty() &&
              std::find(stopTokens.begin(), stopTokens.end(),
                        output.token_ids.back()) != stopTokens.end();
          finishReason = stopped ? "stop" : "length";
        }
        json["finish_reason"] = finishReason;
        // The current engine does not report behavior logprobs. Never
        // substitute top-32 scores or fabricate probabilities for the training
        // client.
        json["logprobs"] = Json::nullValue;
        if (returnIds) {
          json["token_ids"] = Json::Value(Json::arrayValue);
          for (auto token : output.token_ids) json["token_ids"].append(token);
        }
        json["usage"] = response.usage.toJson();
        return Json::writeString(Json::StreamWriterBuilder{}, json);
      };
  handleNonStreaming(
      request, std::move(builder),
      [this, batch](const drogon::HttpResponsePtr& response) {
        if (response->statusCode() != drogon::k200OK) {
          batch->callback(response);
          return;
        }
        const auto result = response->getJsonObject();
        if (!result) {
          batch->callback(errorResponse(drogon::k500InternalServerError,
                                        "Invalid completion result",
                                        "internal_error"));
          return;
        }
        if (isErrorFinishReason((*result)["finish_reason"].asString()) ||
            (*result)["finish_reason"] == "abort") {
          batch->callback(errorResponse(drogon::k500InternalServerError,
                                        "Generation did not complete",
                                        "internal_error"));
          return;
        }
        Json::Value choice = *result;
        batch->response["created"] = choice["created"];
        choice.removeMember("created");
        for (const char* key :
             {"prompt_tokens", "completion_tokens", "total_tokens"}) {
          batch->response["usage"][key] =
              batch->response["usage"][key].asUInt64() +
              choice["usage"][key].asUInt64();
        }
        choice.removeMember("usage");
        choice["index"] = Json::UInt64(batch->next);
        batch->response["choices"].append(std::move(choice));
        if (++batch->next == batch->requests.size()) {
          batch->callback(
              drogon::HttpResponse::newHttpJsonResponse(batch->response));
        } else {
          // Use the existing session scheduler; submit the next prompt after
          // this one releases its slot. This also supports batches larger than
          // the mesh.
          batch->loop->queueInLoop([this, batch] { completeBatch(batch); });
        }
      });
}
}  // namespace tt::api
