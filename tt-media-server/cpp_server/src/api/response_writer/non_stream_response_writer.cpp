// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/response_writer/non_stream_response_writer.hpp"

#include <json/json.h>

#include <utility>

#include "api/error_response.hpp"
#include "domain/llm/chat_completion_response.hpp"
#include "domain/llm/llm_response.hpp"

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
  accumulatedAnswer << choice.text;

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

  llmResponse.choices.push_back(std::move(choice));
  llmResponse.usage = buildUsage();

  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  resp->setBody(builder(llmResponse));

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
    cb(errorResponse(status, message, type));
  }
}

}  // namespace tt::api
