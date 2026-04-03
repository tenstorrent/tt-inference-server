// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <json/json.h>
#include <trantor/net/EventLoop.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>

#include "api/llm_controller.hpp"
#include "config/settings.hpp"
#include "domain/chat_completion_request.hpp"
#include "domain/chat_completion_response.hpp"
#include "profiling/tracy.hpp"
#include "utils/concurrent_queue.hpp"
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
        // Stream not ready yet — buffer for flush when it opens.
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

Json::Value LLMController::errorJson(const std::string& message,
                                     const std::string& type,
                                     const Json::Value& param,
                                     const Json::Value& code) {
  Json::Value error;
  error["object"] = "error";
  error["message"] = message;
  error["type"] = type;
  error["param"] = param;
  error["code"] = code;
  return error;
}

void LLMController::chatCompletions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::chat_completions");

  auto json = req->getJsonObject();
  if (!json) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        errorJson("Invalid JSON body", "invalid_request_error"));
    resp->setStatusCode(drogon::k400BadRequest);
    callback(resp);
    return;
  }

  std::optional<domain::ChatCompletionRequest> chatReqOpt;
  try {
    uint32_t taskId = tt::utils::TaskIDGenerator::generate();
    chatReqOpt =
        domain::ChatCompletionRequest::fromJson(*json, std::move(taskId));
  } catch (const std::exception& e) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        errorJson(std::string("Failed to parse request: ") + e.what(),
                  "invalid_request_error"));
    resp->setStatusCode(drogon::k400BadRequest);
    callback(resp);
    return;
  }

  domain::ChatCompletionRequest& chatReq = *chatReqOpt;

  TT_LOG_INFO("[LLMController] /v1/chat/completions {}", chatReq.toString());

  if (chatReq.messages.empty()) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        errorJson("messages is required and must be a non-empty array",
                  "invalid_request_error", Json::Value("messages")));
    resp->setStatusCode(drogon::k400BadRequest);
    callback(resp);
    return;
  }

  if (!service->isModelReady()) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        errorJson("Model is not ready", "service_unavailable"));
    resp->setStatusCode(drogon::k503ServiceUnavailable);
    callback(resp);
    return;
  }

  auto request = std::make_shared<domain::LLMRequest>(chatReq.toLLMRequest());

  if (request->stream) {
    handleStreaming(request, std::move(callback));
  } else {
    try {
      // Create session if not provided
      if (!request->sessionId.has_value() && sessionManager) {
        auto session = sessionManager->createSession(std::nullopt);
        request->sessionId = session.getSessionId();
      }

      // Save sessionId before moving request
      auto sessionId = request->sessionId;
      request->slotId =
          sessionManager->getSlotIdBySessionId(request->sessionId.value());
      auto startTime = std::chrono::high_resolution_clock::now();
      auto completion = service->submitRequest(std::move(*request));
      auto endTime = std::chrono::high_resolution_clock::now();

      completion.id = "chatcmpl-" + completion.id;

      // Add timing metrics to non-streaming response
      auto totalDuration =
          std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                                startTime);
      if (totalDuration.count() > 0 && completion.usage.completion_tokens > 0) {
        completion.usage.ttft_ms = static_cast<double>(totalDuration.count());
        if (completion.usage.completion_tokens > 1) {
          completion.usage.tps = completion.usage.completion_tokens * 1000.0 /
                                 totalDuration.count();
        }
      }

      // Include sessionId in response if present
      if (sessionId.has_value()) {
        completion.usage.sessionId = sessionId;
      }

      auto chatResponse =
          domain::ChatCompletionResponse::fromLLMResponse(completion);
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      resp->setBody(chatResponse.toJsonString());
      callback(resp);
    } catch (const services::QueueFullException& e) {
      auto resp = drogon::HttpResponse::newHttpJsonResponse(
          errorJson(e.what(), "rate_limit_exceeded"));
      resp->setStatusCode(drogon::k429TooManyRequests);
      callback(resp);
    }
  }
}

void LLMController::handleStreaming(
    std::shared_ptr<domain::LLMRequest> reqPtr,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::handleStreaming");

  bool validSessionFound = false;

  if (reqPtr->sessionId.has_value() && sessionManager) {
    auto slotId =
        sessionManager->getSlotIdBySessionId(reqPtr->sessionId.value());

    if (slotId != tt::services::INVALID_SLOT_ID) {
      reqPtr->slotId = slotId;
      validSessionFound = true;  // Session is valid
    } else {
      TT_LOG_INFO(
          "Received request with non existing session, resetting session id");
      // reset sessionId since it's a stale session
      reqPtr->sessionId.reset();
    }
  }

  // Create session if not provided
  if (!reqPtr->sessionId.has_value() && sessionManager) {
    auto session = sessionManager->createSession(std::nullopt);
    reqPtr->sessionId = session.getSessionId();
    TT_LOG_INFO("[LLMController] Created NEW session: {}",
                reqPtr->sessionId.value());
  }

  if (reqPtr->sessionId.has_value() && sessionManager) {
    reqPtr->slotId =
        sessionManager->getSlotIdBySessionId(reqPtr->sessionId.value());
    TT_LOG_DEBUG(
        "[LLMController] Session: {}, SlotID: {}", reqPtr->sessionId.value(),
        reqPtr->slotId.has_value() ? std::to_string(reqPtr->slotId.value())
                                   : "none");
  } else if (!sessionManager) {
    TT_LOG_WARN("[LLMController] SessionManager not available");
  }

  const std::string completionId =
      "chatcmpl-" + std::to_string(reqPtr->task_id);
  const std::string model = reqPtr->model.value_or("default");
  const int64_t created = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  const bool includeUsage =
      !reqPtr->stream_options.has_value() ||
      reqPtr->stream_options
          ->include_usage;  // Default to true if not specified
  const bool continuousUsage = reqPtr->stream_options.has_value() &&
                               reqPtr->stream_options->continuous_usage_stats;

  auto accumulator = tt::config::enableAccumulatedStreaming()
                         ? std::make_shared<ConcurrentQueue<std::string>>()
                         : nullptr;

  // Request handlers run on the IO thread, so we can capture the loop here.
  trantor::EventLoop* loop = trantor::EventLoop::getEventLoopOfCurrentThread();

  // Shared state. All accesses happen on the IO thread: token callbacks via
  // queueInLoop, and the stream setup lambda via Drogon's async stream
  // machinery.
  auto done = std::make_shared<std::atomic<bool>>(false);
  // stream_ptr is null until the newAsyncStreamResponse lambda runs.
  auto streamPtr = std::make_shared<drogon::ResponseStreamPtr>();
  // Events that arrive before stream_ptr is set are buffered here.
  auto earlyBuffer = std::make_shared<std::vector<std::string>>();
  auto completionTokens = std::make_shared<std::atomic<int>>(0);

  // Timing metrics for TTFT and TPS calculation
  auto startTime =
      std::make_shared<std::chrono::high_resolution_clock::time_point>(
          std::chrono::high_resolution_clock::now());
  auto firstTokenTime = std::make_shared<
      std::optional<std::chrono::high_resolution_clock::time_point>>();
  auto secondTokenTime = std::make_shared<
      std::optional<std::chrono::high_resolution_clock::time_point>>();
  auto firstContentChunk = std::make_shared<std::atomic<bool>>(true);

  // Capture sessionId before submitting to service (request will be moved)
  const std::optional<std::string> capturedSessionId = reqPtr->sessionId;
  // Abort callback: fires when the TCP connection drops mid-stream.
  // Captured by value so it stays alive for the duration of the streaming
  // session. Called on the IO thread from inside sseSend().
  const uint32_t taskId = reqPtr->task_id;
  auto onDisconnect = [taskId, service = this->service, done]() {
    if (!done->exchange(true)) {
      TT_LOG_INFO("[LLMController] Client disconnected, aborting task {}",
                  taskId);
      service->abortRequest(taskId);
    }
  };

  // Submit the streaming request before setting up the HTTP stream so that a
  // full queue throws QueueFullException here, allowing us to return a proper
  // 429.
  auto streamingCallback = [loop, streamPtr, earlyBuffer, done, completionId,
                            model, created, includeUsage, continuousUsage,
                            completionTokens, startTime, firstTokenTime,
                            secondTokenTime, firstContentChunk, reqPtr,
                            capturedSessionId, accumulator,
                            onDisconnect](const domain::LLMStreamChunk& chunk,
                                          bool isFinal) {
    if (done->load()) {
      return;
    }
    if (!chunk.choices.empty()) {
      const int currentTokens = completionTokens->fetch_add(1) + 1;

      auto now = std::chrono::high_resolution_clock::now();
      if (!firstTokenTime->has_value()) {
        *firstTokenTime = now;
      } else if (currentTokens == 2 && !secondTokenTime->has_value()) {
        *secondTokenTime = now;
      }

      std::optional<domain::CompletionUsage> usage;
      if (continuousUsage) {
        usage = domain::CompletionUsage{
            reqPtr->prompt_tokens_count,
            currentTokens,
            currentTokens,
            std::nullopt,
            std::nullopt,
            capturedSessionId,
        };
      }
      auto streamChunk = domain::ChatCompletionStreamChunk::makeContentChunk(
          completionId, model, created, chunk.choices[0], usage);

      std::string sse;
      if (firstContentChunk->exchange(false)) {
        std::optional<domain::CompletionUsage> initialUsage;
        if (continuousUsage) {
          initialUsage = domain::CompletionUsage{reqPtr->prompt_tokens_count,
                                                 0,
                                                 0,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 capturedSessionId};
        }
        auto initialChunk = domain::ChatCompletionStreamChunk::makeInitialChunk(
            completionId, model, created, initialUsage);
        sse = initialChunk.toSSE() + streamChunk.toSSE();
      } else {
        sse = streamChunk.toSSE();
      }

      if (!sse.empty()) {
        sseSend(sse, loop, streamPtr, accumulator, earlyBuffer, onDisconnect);
      }
    }
    if (isFinal) {
      loop->queueInLoop([streamPtr, done, includeUsage, completionId, model,
                         created, completionTokens, startTime, firstTokenTime,
                         secondTokenTime, reqPtr, capturedSessionId,
                         accumulator]() {
        if (!done->exchange(true) && *streamPtr) {
          flushAccumulated(accumulator, streamPtr);
          if (includeUsage) {
            const int tokens = completionTokens->load();

            TT_LOG_DEBUG(
                "[LLMController] Creating final usage - capturedSessionId: {}",
                capturedSessionId.has_value()
                    ? ("'" + capturedSessionId.value() + "'")
                    : "none");

            domain::CompletionUsage usage{reqPtr->prompt_tokens_count,
                                          tokens,
                                          tokens,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt};

            if (firstTokenTime->has_value()) {
              auto ttftDuration =
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      firstTokenTime->value() - *startTime);
              usage.ttft_ms =
                  std::round(static_cast<double>(ttftDuration.count()) / 10.0) /
                  100.0;
            }

            if (tokens > 1) {
              auto finalTime = std::chrono::high_resolution_clock::now();
              auto baseTime = secondTokenTime->has_value()
                                  ? secondTokenTime->value()
                                  : firstTokenTime->value();
              auto totalDuration =
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      finalTime - baseTime);
              if (totalDuration.count() > 0) {
                auto timeSeconds =
                    static_cast<double>(totalDuration.count()) / 1000000.0;
                usage.tps =
                    std::round((tokens - 1) / timeSeconds * 1000.0) / 1000.0;
                TT_LOG_DEBUG("[LLMController] Final TPS: {} tokens/sec",
                             usage.tps.value());
              }
            }

            // Include sessionId in usage if present
            if (capturedSessionId.has_value()) {
              usage.sessionId = capturedSessionId;
              TT_LOG_DEBUG("[LLMController] Set usage.sessionId to: '{}'",
                           usage.sessionId.value());
            } else {
              TT_LOG_WARN(
                  "[LLMController] capturedSessionId is empty, cannot set "
                  "usage.sessionId");
            }

            (*streamPtr)
                ->send(domain::ChatCompletionStreamChunk::makeUsageChunk(
                           completionId, model, created, usage)
                           .toSSE());
          }
          (*streamPtr)->send("data: [DONE]\n\n");
          (*streamPtr)->close();
        }
      });
    }
  };
  try {
    if (tt::config::llmMode() == tt::config::LLMMode::REGULAR) {
      service->submitStreamingRequest(*reqPtr, streamingCallback);
    } else if (tt::config::llmMode() == tt::config::LLMMode::DECODE_ONLY) {
      // tokenize right away
      service->preProcess(*reqPtr);
      if (shouldDoPrefillOnDecode(*reqPtr, validSessionFound)) {
        TT_LOG_DEBUG(
            "[LLMController] Using prefill on decode for sessionId: {}",
            reqPtr->sessionId.value_or("none"));
        service->submitStreamingRequest(*reqPtr, streamingCallback);
      } else {
        TT_LOG_DEBUG(
            "[LLMController] Using disaggregated prefill for request with "
            "sessionId: {}",
            reqPtr->sessionId.value_or("none"));
        disaggregationService->handleStreamingRequest(*reqPtr,
                                                      streamingCallback);
      }
    } else {
      throw std::runtime_error(
          "[LLMController] LLM Mode must be regular or decode only to handle "
          "streaming requests via HTTP");
    }
  } catch (const services::QueueFullException& e) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        errorJson(e.what(), "rate_limit_exceeded"));
    resp->setStatusCode(drogon::k429TooManyRequests);
    callback(resp);
    return;
  }

  auto resp = drogon::HttpResponse::newAsyncStreamResponse(
      [streamPtr, earlyBuffer, done](drogon::ResponseStreamPtr stream) mutable {
        *streamPtr = std::move(stream);
        for (const auto& event : *earlyBuffer) {
          (*streamPtr)->send(event);
        }
        earlyBuffer->clear();
        if (done->load()) {
          (*streamPtr)->close();
        }
      });

  resp->setContentTypeString("text/event-stream");
  resp->addHeader("Cache-Control", "no-cache");
  resp->addHeader("Connection", "keep-alive");
  resp->addHeader("X-Accel-Buffering", "no");

  callback(resp);
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
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        errorJson(e.what(), "internal_error"));
    resp->setStatusCode(drogon::k500InternalServerError);
    callback(resp);
  }
}

void LLMController::closeSession(
    const drogon::HttpRequestPtr& req,
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
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        errorJson("Session not found", "not_found"));
    resp->setStatusCode(drogon::k404NotFound);
    callback(resp);
  }
}

void LLMController::getSlotId(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& sessionId) const {
  uint32_t slotId = sessionManager->getSlotIdBySessionId(sessionId);

  if (slotId == tt::services::INVALID_SLOT_ID) {
    // Check if session exists at all
    auto session = sessionManager->getSession(sessionId);
    if (!session.has_value()) {
      auto resp = drogon::HttpResponse::newHttpJsonResponse(
          errorJson("Session not found", "not_found"));
      resp->setStatusCode(drogon::k404NotFound);
      callback(resp);
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
