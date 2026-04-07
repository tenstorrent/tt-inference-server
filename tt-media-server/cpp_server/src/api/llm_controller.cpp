// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <json/json.h>
#include <trantor/net/EventLoop.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>

#include "api/error_response.hpp"
#include "api/llm_controller.hpp"
#include "config/settings.hpp"
#include "domain/chat_completion_request.hpp"
#include "domain/chat_completion_response.hpp"
#include "profiling/tracy.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/service_container.hpp"

namespace {

void sseSend(const std::string& sse, trantor::EventLoop* loop,
             const std::shared_ptr<drogon::ResponseStreamPtr>& streamPtr,
             const std::shared_ptr<ConcurrentQueue<std::string>>& accumulator,
             const std::shared_ptr<std::vector<std::string>>& earlyBuffer,
             std::function<void()> onDisconnect = nullptr) {
  if (!accumulator) {
    loop->queueInLoop([streamPtr, earlyBuffer, sse,
                       onDisconnect = std::move(onDisconnect)]() {
      if (*streamPtr) {
        bool ok = (*streamPtr)->send(sse);
        if (!ok && onDisconnect) onDisconnect();
      } else if (earlyBuffer) {
        earlyBuffer->push_back(sse);
      }
    });
    return;
  }
  accumulator->push(sse);
  if (accumulator->size() >= tt::config::maxAccumulatedTokens()) {
    auto accumulated = accumulator->drain();
    std::string batch;
    for (auto& s : accumulated) batch.append(s);
    loop->queueInLoop([streamPtr, earlyBuffer, batch = std::move(batch),
                       onDisconnect = std::move(onDisconnect)]() {
      if (*streamPtr) {
        bool ok = (*streamPtr)->send(batch);
        if (!ok && onDisconnect) onDisconnect();
      } else if (earlyBuffer) {
        earlyBuffer->push_back(batch);
      }
    });
  }
}

void flushAccumulated(
    const std::shared_ptr<ConcurrentQueue<std::string>>& accumulator,
    const std::shared_ptr<drogon::ResponseStreamPtr>& streamPtr) {
  if (!accumulator) return;
  auto accumulated = accumulator->drain();
  if (!accumulated.empty()) {
    std::string batch;
    for (auto& s : accumulated) batch.append(s);
    if (*streamPtr) (*streamPtr)->send(batch);
  }
}

// Bundles all shared state for a streaming SSE session, replacing 17+
// individual shared_ptr captures with a single shared_ptr<StreamContext>.
struct StreamContext {
  // Stream infrastructure
  trantor::EventLoop* loop;
  std::shared_ptr<drogon::ResponseStreamPtr> streamPtr =
      std::make_shared<drogon::ResponseStreamPtr>();
  std::shared_ptr<std::vector<std::string>> earlyBuffer =
      std::make_shared<std::vector<std::string>>();
  std::shared_ptr<ConcurrentQueue<std::string>> accumulator;
  std::atomic<bool> done{false};

  // Token counting + timing
  std::atomic<int> completionTokens{0};
  std::chrono::high_resolution_clock::time_point startTime =
      std::chrono::high_resolution_clock::now();
  std::optional<std::chrono::high_resolution_clock::time_point> firstTokenTime;
  std::optional<std::chrono::high_resolution_clock::time_point> secondTokenTime;
  std::atomic<bool> firstContentChunk{true};

  // Request metadata (immutable after construction)
  std::string completionId;
  std::string model;
  int64_t created;
  bool includeUsage;
  bool continuousUsage;
  int promptTokensCount;
  std::optional<std::string> sessionId;
  uint32_t taskId;

  // Dependencies
  std::shared_ptr<tt::services::LLMService> service;
  std::shared_ptr<tt::services::SessionManager> sessionManager;
};

tt::domain::CompletionUsage buildFinalUsage(const StreamContext& ctx) {
  const int tokens = ctx.completionTokens.load();

  tt::domain::CompletionUsage usage{
      ctx.promptTokensCount, tokens,       tokens,
      std::nullopt,          std::nullopt, std::nullopt};

  if (ctx.firstTokenTime.has_value()) {
    auto ttftUs = std::chrono::duration_cast<std::chrono::microseconds>(
        ctx.firstTokenTime.value() - ctx.startTime);
    usage.ttft_ms =
        std::round(static_cast<double>(ttftUs.count()) / 10.0) / 100.0;
  }

  if (tokens > 1 && ctx.firstTokenTime.has_value()) {
    auto finalTime = std::chrono::high_resolution_clock::now();
    auto baseTime = ctx.secondTokenTime.value_or(ctx.firstTokenTime.value());
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(
        finalTime - baseTime);
    if (totalUs.count() > 0) {
      auto secs = static_cast<double>(totalUs.count()) / 1000000.0;
      usage.tps = std::round((tokens - 1) / secs * 1000.0) / 1000.0;
    }
  }

  if (ctx.sessionId.has_value()) {
    usage.sessionId = ctx.sessionId;
  }
  return usage;
}

void finalizeStream(const std::shared_ptr<StreamContext>& ctx) {
  ctx->loop->queueInLoop([ctx]() {
    if (!ctx->done.exchange(true) && *ctx->streamPtr) {
      flushAccumulated(ctx->accumulator, ctx->streamPtr);

      if (ctx->includeUsage) {
        auto usage = buildFinalUsage(*ctx);
        (*ctx->streamPtr)
            ->send(tt::domain::ChatCompletionStreamChunk::makeUsageChunk(
                       ctx->completionId, ctx->model, ctx->created, usage)
                       .toSSE());
      }

      (*ctx->streamPtr)->send("data: [DONE]\n\n");
      (*ctx->streamPtr)->close();

      if (ctx->sessionId.has_value() && ctx->sessionManager) {
        ctx->sessionManager->setSessionInFlight(ctx->sessionId.value(), false);
      }
    }
  });
}

void handleTokenChunk(const std::shared_ptr<StreamContext>& ctx,
                      const tt::domain::LLMStreamChunk& chunk,
                      const std::function<void()>& onDisconnect) {
  const int currentTokens = ctx->completionTokens.fetch_add(1) + 1;

  auto now = std::chrono::high_resolution_clock::now();
  if (!ctx->firstTokenTime.has_value()) {
    ctx->firstTokenTime = now;
  } else if (currentTokens == 2 && !ctx->secondTokenTime.has_value()) {
    ctx->secondTokenTime = now;
  }

  std::optional<tt::domain::CompletionUsage> usage;
  if (ctx->continuousUsage) {
    usage = tt::domain::CompletionUsage{ctx->promptTokensCount, currentTokens,
                                        currentTokens,          std::nullopt,
                                        std::nullopt,           ctx->sessionId};
  }

  auto streamChunk = tt::domain::ChatCompletionStreamChunk::makeContentChunk(
      ctx->completionId, ctx->model, ctx->created, chunk.choices[0], usage);

  std::string sse;
  if (ctx->firstContentChunk.exchange(false)) {
    std::optional<tt::domain::CompletionUsage> initialUsage;
    if (ctx->continuousUsage) {
      initialUsage = tt::domain::CompletionUsage{
          ctx->promptTokensCount, 0, 0, std::nullopt, std::nullopt,
          ctx->sessionId};
    }
    auto initialChunk = tt::domain::ChatCompletionStreamChunk::makeInitialChunk(
        ctx->completionId, ctx->model, ctx->created, initialUsage);
    sse = initialChunk.toSSE() + streamChunk.toSSE();
  } else {
    sse = streamChunk.toSSE();
  }

  if (!sse.empty()) {
    sseSend(sse, ctx->loop, ctx->streamPtr, ctx->accumulator, ctx->earlyBuffer,
            onDisconnect);
  }
}

}  // anonymous namespace

namespace tt::api {

LLMController::LLMController() {
  if (!tt::config::isLlmServiceEnabled()) {
    TT_LOG_INFO(
        "[LLMController] Skipping initialization (TT_model_SERVICE != llm)");
    return;
  }

  const auto& c = tt::utils::ServiceContainer::instance();
  service = c.llm();
  disaggregationService = c.disaggregation();
  sessionManager = c.sessionManager();

  if (!service) {
    throw std::runtime_error(
        "[LLMController] LLM service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[LLMController] Initialized (service already started)");
}

void LLMController::resolveSessionAsync(
    std::shared_ptr<domain::LLMRequest> reqPtr,
    std::function<void(SessionInfo)> onResolved,
    std::function<void(const std::string&)> onError) const {
  SessionInfo info;

  if (reqPtr->sessionId.has_value() && sessionManager) {
    auto slotId =
        sessionManager->acquireSessionSlot(reqPtr->sessionId.value());
    if (slotId != services::INVALID_SLOT_ID) {
      reqPtr->slotId = slotId;
      info.validSessionFound = true;
      onResolved(info);
      return;
    }
    TT_LOG_INFO(
        "Received request with non existing session, resetting session id");
    reqPtr->sessionId.reset();
  }

  if (!reqPtr->sessionId.has_value() && sessionManager) {
    auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
    sessionManager->createSessionAsync(
        [reqPtr, onResolved, info](domain::Session session) mutable {
          reqPtr->sessionId = session.getSessionId();
          reqPtr->slotId = session.getSlotId();
          onResolved(info);
        },
        [onError = std::move(onError)](const std::string& err) {
          onError(err);
        },
        loop, /*inFlight=*/true);
    return;
  }

  if (!sessionManager) {
    TT_LOG_WARN("[LLMController] SessionManager not available");
  }

  onResolved(info);
}

void LLMController::chatCompletions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::chat_completions");

  auto json = req->getJsonObject();
  if (!json) {
    callback(errorResponse(drogon::k400BadRequest, "Invalid JSON body",
                           "invalid_request_error"));
    return;
  }

  std::optional<domain::ChatCompletionRequest> chatReqOpt;
  try {
    uint32_t taskId = tt::utils::TaskIDGenerator::generate();
    chatReqOpt =
        domain::ChatCompletionRequest::fromJson(*json, std::move(taskId));
  } catch (const std::exception& e) {
    callback(errorResponse(drogon::k400BadRequest,
                           std::string("Failed to parse request: ") + e.what(),
                           "invalid_request_error"));
    return;
  }

  domain::ChatCompletionRequest& chatReq = *chatReqOpt;

  TT_LOG_INFO("[LLMController] /v1/chat/completions {}", chatReq.toString());

  if (chatReq.messages.empty()) {
    callback(errorResponse(drogon::k400BadRequest,
                           "messages is required and must be a non-empty array",
                           "invalid_request_error", Json::Value("messages")));
    return;
  }

  if (!service->isModelReady()) {
    callback(errorResponse(drogon::k503ServiceUnavailable, "Model is not ready",
                           "service_unavailable"));
    return;
  }

  auto request = std::make_shared<domain::LLMRequest>(chatReq.toLLMRequest());

  if (request->stream) {
    handleStreaming(request, std::move(callback));
  } else {
    auto cb = std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
        std::move(callback));

    resolveSessionAsync(
        request,
        [this, request, cb](SessionInfo /*sessionInfo*/) {
          try {
            auto sessionId = request->sessionId;
            auto startTime = std::chrono::high_resolution_clock::now();
            auto completion = service->submitRequest(std::move(*request));
            auto endTime = std::chrono::high_resolution_clock::now();

            completion.id = "chatcmpl-" + completion.id;

            auto totalDuration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime);
            if (totalDuration.count() > 0 &&
                completion.usage.completion_tokens > 0) {
              completion.usage.ttft_ms =
                  static_cast<double>(totalDuration.count());
              if (completion.usage.completion_tokens > 1) {
                completion.usage.tps =
                    completion.usage.completion_tokens * 1000.0 /
                    totalDuration.count();
              }
            }

            if (sessionId.has_value()) {
              completion.usage.sessionId = sessionId;
            }

            auto chatResponse =
                domain::ChatCompletionResponse::fromLLMResponse(completion);
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            resp->setBody(chatResponse.toJsonString());

            if (sessionId.has_value() && sessionManager) {
              sessionManager->setSessionInFlight(sessionId.value(), false);
            }

            (*cb)(resp);
          } catch (const services::QueueFullException& e) {
            auto sessionId = request->sessionId;
            if (sessionId.has_value() && sessionManager) {
              sessionManager->setSessionInFlight(sessionId.value(), false);
            }
            (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                                "rate_limit_exceeded"));
          } catch (const std::runtime_error& e) {
            auto sessionId = request->sessionId;
            if (sessionId.has_value() && sessionManager) {
              sessionManager->setSessionInFlight(sessionId.value(), false);
            }
            std::string errMsg = e.what();
            if (errMsg.find("Failed to allocate memory slot") !=
                    std::string::npos ||
                errMsg.find("memory resources") != std::string::npos) {
              TT_LOG_ERROR("[LLMController] Session creation failed: {}",
                           errMsg);
              (*cb)(errorResponse(
                  drogon::k503ServiceUnavailable,
                  std::string("Failed to allocate memory resources: ") + errMsg,
                  "service_unavailable"));
              return;
            }
            TT_LOG_ERROR("[LLMController] Unexpected error: {}", errMsg);
            (*cb)(errorResponse(drogon::k500InternalServerError, errMsg,
                                "internal_error"));
          }
        },
        [cb](const std::string& error) {
          TT_LOG_ERROR("[LLMController] Session creation failed: {}", error);
          (*cb)(errorResponse(
              drogon::k503ServiceUnavailable,
              std::string("Failed to allocate memory resources: ") + error,
              "service_unavailable"));
        });
  }
}

void LLMController::handleStreaming(
    std::shared_ptr<domain::LLMRequest> reqPtr,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  auto cb =
      std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

  resolveSessionAsync(
      reqPtr,
      [this, reqPtr, cb](SessionInfo sessionInfo) {
        auto ctx = std::make_shared<StreamContext>();
        ctx->loop = trantor::EventLoop::getEventLoopOfCurrentThread();
        ctx->accumulator =
            tt::config::enableAccumulatedStreaming()
                ? std::make_shared<ConcurrentQueue<std::string>>()
                : nullptr;
        ctx->completionId = "chatcmpl-" + std::to_string(reqPtr->task_id);
        ctx->model = reqPtr->model.value_or("default");
        ctx->created = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
        ctx->includeUsage = !reqPtr->stream_options.has_value() ||
                            reqPtr->stream_options->include_usage;
        ctx->continuousUsage = reqPtr->stream_options.has_value() &&
                               reqPtr->stream_options->continuous_usage_stats;
        ctx->promptTokensCount = reqPtr->prompt_tokens_count;
        ctx->sessionId = reqPtr->sessionId;
        ctx->taskId = reqPtr->task_id;
        ctx->service = service;
        ctx->sessionManager = sessionManager;

        auto onDisconnect = [ctx]() {
          if (!ctx->done.exchange(true)) {
            TT_LOG_INFO(
                "[LLMController] Client disconnected, aborting task {}",
                ctx->taskId);
            ctx->service->abortRequest(ctx->taskId);
            if (ctx->sessionId.has_value() && ctx->sessionManager) {
              ctx->sessionManager->setSessionInFlight(ctx->sessionId.value(),
                                                      false);
            }
          }
        };

        auto streamingCallback =
            [ctx,
             onDisconnect](const domain::LLMStreamChunk& chunk, bool isFinal) {
              if (ctx->done.load()) return;
              if (!chunk.choices.empty())
                handleTokenChunk(ctx, chunk, onDisconnect);
              if (isFinal) finalizeStream(ctx);
            };

        try {
          if (tt::config::llmMode() == tt::config::LLMMode::REGULAR) {
            service->submitStreamingRequest(*reqPtr, streamingCallback);
          } else if (tt::config::llmMode() ==
                     tt::config::LLMMode::DECODE_ONLY) {
            service->preProcess(*reqPtr);
            if (shouldDoPrefillOnDecode(*reqPtr,
                                        sessionInfo.validSessionFound)) {
              TT_LOG_DEBUG(
                  "[LLMController] Using prefill on decode for sessionId: {}",
                  reqPtr->sessionId.value_or("none"));
              service->submitStreamingRequest(*reqPtr, streamingCallback);
            } else {
              TT_LOG_DEBUG(
                  "[LLMController] Using disaggregated prefill for request "
                  "with sessionId: {}",
                  reqPtr->sessionId.value_or("none"));
              disaggregationService->handleStreamingRequest(*reqPtr,
                                                            streamingCallback);
            }
          } else {
            (*cb)(errorResponse(
                drogon::k500InternalServerError,
                "LLM Mode must be regular or decode only for streaming",
                "internal_error"));
            return;
          }
        } catch (const services::QueueFullException& e) {
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        }

        auto resp = drogon::HttpResponse::newAsyncStreamResponse(
            [ctx](drogon::ResponseStreamPtr stream) mutable {
              *ctx->streamPtr = std::move(stream);
              for (const auto& event : *ctx->earlyBuffer) {
                (*ctx->streamPtr)->send(event);
              }
              ctx->earlyBuffer->clear();
              if (ctx->done.load()) {
                (*ctx->streamPtr)->close();
              }
            });

        resp->setContentTypeString("text/event-stream");
        resp->addHeader("Cache-Control", "no-cache");
        resp->addHeader("Connection", "keep-alive");
        resp->addHeader("X-Accel-Buffering", "no");

        (*cb)(resp);
      },
      [cb](const std::string& error) {
        TT_LOG_ERROR("[LLMController] Failed to create session: {}", error);
        (*cb)(errorResponse(
            drogon::k503ServiceUnavailable,
            std::string("Failed to allocate memory resources: ") + error,
            "service_unavailable"));
      });
}

bool LLMController::shouldDoPrefillOnDecode(const domain::LLMRequest& request,
                                            bool validSessionFound) const {
  // for valid sessions always do prefill on decode
  if (validSessionFound) {
    return true;
  }

  // Check if the number of prompt tokens exceeds the threshold
  // If tokens are below the threshold, prefill on decode server
  const size_t maxTokens = tt::config::maxTokensToPrefillOnDecode();
  const size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  return promptTokens < maxTokens;
}

void LLMController::createSession(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  try {
    std::optional<uint32_t> slotId;

    // Try to parse request body for optional slotId
    if (req->getBody().length() > 0) {
      Json::Value requestBody;
      Json::CharReaderBuilder builder;
      std::istringstream bodyStream(std::string(req->getBody()));
      std::string errs;

      if (Json::parseFromStream(builder, bodyStream, &requestBody, &errs)) {
        if (requestBody.isMember("slot_id") &&
            requestBody["slot_id"].isUInt()) {
          slotId = requestBody["slot_id"].asUInt();
        }
      }
    }

    auto session = sessionManager->createSession(slotId);

    Json::Value response = session.toJson();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k201Created);
    callback(resp);
  } catch (const std::exception& e) {
    callback(errorResponse(drogon::k500InternalServerError, e.what(),
                           "internal_error"));
  }
}

void LLMController::closeSession(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& sessionId) const {
  bool success = sessionManager->closeSession(sessionId);

  if (success) {
    Json::Value response;
    response["success"] = true;
    response["message"] = "Session closed";
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    callback(resp);
  } else {
    callback(
        errorResponse(drogon::k404NotFound, "Session not found", "not_found"));
  }
}

void LLMController::getSlotId(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& sessionId) const {
  uint32_t slotId = sessionManager->getSlotIdBySessionId(sessionId);

  if (slotId == tt::services::INVALID_SLOT_ID) {
    // Check if session exists at all
    auto session = sessionManager->getSession(sessionId);
    if (!session.has_value()) {
      callback(errorResponse(drogon::k404NotFound, "Session not found",
                             "not_found"));
      return;
    }
  }

  Json::Value response;
  response["session_id"] = sessionId;
  response["slot_id"] = slotId;
  auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
  callback(resp);
}

}  // namespace tt::api
