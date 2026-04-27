// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/accumulating_response_builder.hpp"

#include <cmath>
#include <utility>

#include "api/error_response.hpp"
#include "domain/chat_completion_response.hpp"
#include "utils/logger.hpp"

namespace tt::api {

AccumulatingResponseBuilder::AccumulatingResponseBuilder(
    AccumulatingResponseParams params, HttpCallback httpCallback)
    : params(std::move(params)), httpCallback(std::move(httpCallback)) {}

std::shared_ptr<AccumulatingResponseBuilder>
AccumulatingResponseBuilder::create(AccumulatingResponseParams params,
                                    HttpCallback httpCallback) {
  return std::shared_ptr<AccumulatingResponseBuilder>(
      new AccumulatingResponseBuilder(std::move(params),
                                      std::move(httpCallback)));
}

void AccumulatingResponseBuilder::handleTokenChunk(
    const domain::LLMStreamChunk& chunk) {
  if (done.load()) return;
  if (chunk.choices.empty()) return;

  // Streaming callbacks for a single task arrive serialized from the LLMService
  // consumer thread (or the disaggregation socket thread), so plain access is
  // safe between handleTokenChunk and finalize on the success path. abort
  // also flows through this same callback (LLMService::abortRequest invokes
  // the registered streaming callback with isFinal=true), so finalize handles
  // the abort case uniformly.

  const auto& choice = chunk.choices[0];
  if (choice.reasoning.has_value()) {
    accumulatedReasoning.append(choice.reasoning.value());
  }
  accumulatedAnswer.append(choice.text);
  ++completionTokens;

  auto now = std::chrono::high_resolution_clock::now();
  if (!firstTokenTime.has_value()) {
    firstTokenTime = now;
  } else if (completionTokens == 2 && !secondTokenTime.has_value()) {
    secondTokenTime = now;
  }

  if (choice.finish_reason.has_value()) {
    finishReason = choice.finish_reason.value();
  }
}

domain::CompletionUsage AccumulatingResponseBuilder::buildFinalUsage() const {
  const int totalTokens = params.promptTokensCount + completionTokens;
  domain::CompletionUsage usage{params.promptTokensCount,
                                completionTokens,
                                totalTokens,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt};

  if (firstTokenTime.has_value()) {
    auto ttftUs = std::chrono::duration_cast<std::chrono::microseconds>(
        firstTokenTime.value() - startTime);
    usage.ttft_ms =
        std::round(static_cast<double>(ttftUs.count()) / 10.0) / 100.0;
  }

  if (completionTokens > 1 && firstTokenTime.has_value()) {
    auto finalTime = std::chrono::high_resolution_clock::now();
    auto baseTime = secondTokenTime.value_or(firstTokenTime.value());
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(
        finalTime - baseTime);
    if (totalUs.count() > 0) {
      auto secs = static_cast<double>(totalUs.count()) / 1000000.0;
      usage.tps =
          std::round((completionTokens - 1) / secs * 1000.0) / 1000.0;
    }
  }

  if (params.sessionId.has_value()) {
    usage.sessionId = params.sessionId;
  }
  return usage;
}

void AccumulatingResponseBuilder::releaseInFlight() {
  if (params.sessionId.has_value() && params.sessionManager) {
    params.sessionManager->releaseInFlight(params.sessionId.value());
  }
}

void AccumulatingResponseBuilder::finalize() {
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

  llmResponse.usage = buildFinalUsage();

  // Tool-call parsing + reasoning strip (non-streaming only). Mirrors the
  // semantics of LLMService::submitRequest's postProcess step.
  if (params.service) {
    try {
      params.service->finalizeResponse(llmResponse);
    } catch (const std::exception& e) {
      TT_LOG_WARN("[AccumulatingResponseBuilder] postProcess failed: {}",
                  e.what());
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

void AccumulatingResponseBuilder::sendError(drogon::HttpStatusCode status,
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
