// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/response_writer/non_stream_response_writer.hpp"

#include <utility>

#include "api/error_response.hpp"
#include "domain/chat_completion_response.hpp"
#include "domain/llm_response.hpp"
#include "utils/logger.hpp"

namespace tt::api {

NonStreamResponseWriter::NonStreamResponseWriter(ResponseWriterParams params,
                                                 HttpCallback httpCallback)
    : ResponseWriter(std::move(params)),
      httpCallback(std::move(httpCallback)) {}

std::shared_ptr<NonStreamResponseWriter> NonStreamResponseWriter::create(
    ResponseWriterParams params, HttpCallback httpCallback) {
  return std::shared_ptr<NonStreamResponseWriter>(
      new NonStreamResponseWriter(std::move(params), std::move(httpCallback)));
}

void NonStreamResponseWriter::handleTokenChunk(
    const domain::LLMStreamChunk& chunk) {
  if (done.load()) return;
  if (chunk.choices.empty()) return;

  const auto& choice = chunk.choices[0];
  if (choice.reasoning.has_value()) {
    accumulatedReasoning.append(choice.reasoning.value());
  }
  accumulatedAnswer.append(choice.text);

  noteToken();

  if (choice.finish_reason.has_value()) {
    finishReason = choice.finish_reason.value();
  }
}

void NonStreamResponseWriter::finalize() {
  if (done.exchange(true)) return;

  domain::LLMResponse llmResponse{params.taskId};
  llmResponse.id = params.completionId;
  llmResponse.model = params.model;
  llmResponse.created = params.created;

  domain::LLMChoice choice;
  choice.index = 0;
  choice.text = std::move(accumulatedAnswer);
  choice.reasoning =
      accumulatedReasoning.empty()
          ? std::nullopt
          : std::optional<std::string>(std::move(accumulatedReasoning));
  choice.finish_reason = finishReason;
  llmResponse.choices.push_back(std::move(choice));

  llmResponse.usage = buildUsage();

  if (params.service) {
    try {
      params.service->finalizeResponse(llmResponse);
    } catch (const std::exception& e) {
      TT_LOG_WARN("[NonStreamResponseWriter] postProcess failed: {}", e.what());
    }
  }

  auto chatResponse =
      domain::ChatCompletionResponse::fromLLMResponse(llmResponse);

  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  resp->setBody(chatResponse.toJsonString());

  releaseInFlight();

  if (httpCallback) {
    auto cb = std::move(httpCallback);
    cb(resp);
  }
}

void NonStreamResponseWriter::sendError(drogon::HttpStatusCode status,
                                        const std::string& message,
                                        const std::string& type) {
  if (done.exchange(true)) return;
  releaseInFlight();
  if (httpCallback) {
    auto cb = std::move(httpCallback);
    cb(errorResponse(status, message, type));
  }
}

}  // namespace tt::api
