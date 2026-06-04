// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/response_writer/non_stream_response_writer.hpp"

#include <json/json.h>

#include <utility>

#include "api/error_response.hpp"
#include "domain/llm/chat_completion_response.hpp"
#include "domain/llm/llm_response.hpp"
#include "utils/logger.hpp"

namespace tt::api {

using namespace tt::domain::llm;

namespace {

std::string defaultChatCompletionBuilder(const LLMResponse& response) {
  return ChatCompletionResponse::fromLLMResponse(response).toJsonString();
}

}  // namespace

NonStreamResponseWriter::NonStreamResponseWriter(ResponseWriterParams params,
                                                 HttpCallback httpCallback,
                                                 ResponseBuilder builder)
    : ResponseWriter(std::move(params)),
      httpCallback(std::move(httpCallback)),
      builder(std::move(builder)) {}

std::shared_ptr<NonStreamResponseWriter> NonStreamResponseWriter::create(
    ResponseWriterParams params, HttpCallback httpCallback) {
  return create(std::move(params), std::move(httpCallback),
                &defaultChatCompletionBuilder);
}

std::shared_ptr<NonStreamResponseWriter> NonStreamResponseWriter::create(
    ResponseWriterParams params, HttpCallback httpCallback,
    ResponseBuilder builder) {
  if (!builder) {
    builder = &defaultChatCompletionBuilder;
  }
  return std::shared_ptr<NonStreamResponseWriter>(new NonStreamResponseWriter(
      std::move(params), std::move(httpCallback), std::move(builder)));
}

void NonStreamResponseWriter::handleTokenChunk(const LLMStreamChunk& chunk) {
  if (done.load()) return;
  if (chunk.choices.empty()) return;

  const auto& choice = chunk.choices[0];
  if (choice.reasoning.has_value()) {
    accumulatedReasoning << choice.reasoning.value();
  }
  accumulatedAnswer << choice.text;

  // Accumulate tool call data from streaming deltas
  if (choice.tool_calls.has_value()) {
    const auto& toolCallsJson = choice.tool_calls.value();
    if (toolCallsJson.isArray()) {
      for (const auto& toolCallDelta : toolCallsJson) {
        if (!toolCallDelta.isMember("index")) continue;

        int index = toolCallDelta["index"].asInt();

        // Ensure vector is large enough
        while (static_cast<int>(accumulatedToolCalls.size()) <= index) {
          accumulatedToolCalls.emplace_back();
        }

        auto& accumulated = accumulatedToolCalls[index];

        // Capture id and name on first delta (TOOL_CALL_START)
        if (toolCallDelta.isMember("id")) {
          accumulated.id = toolCallDelta["id"].asString();
        }
        if (toolCallDelta.isMember("function")) {
          const auto& func = toolCallDelta["function"];
          if (func.isMember("name") && !func["name"].asString().empty()) {
            accumulated.name = func["name"].asString();
          }
          if (func.isMember("arguments")) {
            accumulated.arguments << func["arguments"].asString();
          }
        }
      }
    }
  }

  noteToken(choice);

  if (choice.finish_reason.has_value()) {
    finishReason = choice.finish_reason.value();
  }
}

void NonStreamResponseWriter::finalize() {
  if (done.exchange(true)) return;

  LLMResponse llmResponse{params.taskId};
  llmResponse.id = params.completionId;
  llmResponse.model = params.model;
  llmResponse.created = params.created;

  LLMChoice choice;
  choice.index = 0;
  choice.reasoning =
      accumulatedReasoning.tellp() == 0
          ? std::nullopt
          : std::optional<std::string>(accumulatedReasoning.str());

  // Build tool_calls if any were accumulated
  if (!accumulatedToolCalls.empty()) {
    Json::Value toolCallsArray(Json::arrayValue);
    for (const auto& tc : accumulatedToolCalls) {
      if (tc.id.empty() && tc.name.empty()) continue;

      Json::Value toolCall;
      toolCall["id"] = tc.id;
      toolCall["type"] = "function";
      toolCall["function"]["name"] = tc.name;
      toolCall["function"]["arguments"] = tc.arguments.str();
      toolCallsArray.append(toolCall);
    }

    if (!toolCallsArray.empty()) {
      choice.tool_calls = toolCallsArray;
      choice.text = "";
      choice.finish_reason = "tool_calls";
    } else {
      choice.text = accumulatedAnswer.str();
      choice.finish_reason = finishReason;
    }
  } else {
    choice.text = accumulatedAnswer.str();
    choice.finish_reason = finishReason;
  }

  llmResponse.choices.push_back(std::move(choice));
  llmResponse.usage = buildUsage();

  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  resp->setBody(builder(llmResponse));
  if (!params.traceId.empty()) {
    resp->addHeader("X-Request-Id", params.traceId);
  }

  if (httpCallback) {
    auto cb = std::move(httpCallback);
    cb(resp);
  }
}

void NonStreamResponseWriter::sendError(drogon::HttpStatusCode status,
                                        const std::string& message,
                                        const std::string& type) {
  if (done.exchange(true)) return;
  if (params.onSessionRelease) params.onSessionRelease();
  if (httpCallback) {
    auto cb = std::move(httpCallback);
    cb(withRequestId(errorResponse(status, message, type), params.traceId));
  }
}

}  // namespace tt::api
