// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/llm_controller.hpp"

#include <json/json.h>

#include <chrono>
#include <functional>
#include <memory>
#include <optional>

#include "api/error_response.hpp"
#include "api/sse_stream_writer.hpp"
#include "config/settings.hpp"
#include "domain/chat_completion_request.hpp"
#include "domain/chat_completion_response.hpp"
#include "domain/models_response.hpp"
#include "profiling/tracy.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/service_container.hpp"

namespace tt::api {

void LLMController::models(
    const drogon::HttpRequestPtr& _,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  domain::ModelsResponse response;
  response.data.push_back({toString(tt::config::model())});
  auto resp = drogon::HttpResponse::newHttpJsonResponse(response.toJson());
  callback(resp);
}

LLMController::LLMController() {
  if (!tt::config::isLlmServiceEnabled()) {
    TT_LOG_INFO(
        "[LLMController] Skipping initialization (TT_model_SERVICE != llm)");
    return;
  }

  tt::config::model();

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

void LLMController::resolveSession(
    std::shared_ptr<domain::LLMRequest> req, trantor::EventLoop* loop,
    std::function<void(SessionInfo)> onResolved,
    std::function<void(const SessionError&)> onError) const {
  SessionInfo info;

  // New hash-based routing: try to find a session by prefix hash
  if (req->hasPriorTurn && req->lookupHash.has_value() && sessionManager) {
    try {
      auto slotId = sessionManager->tryAcquireByPrefixHash(*req->lookupHash);

      if (slotId.has_value()) {
        // HIT: found matching session, send delta only
        TT_LOG_INFO(
            "[LLMController] Prefix cache HIT: hash={}, slotId={}, sending delta",
            *req->lookupHash, *slotId);
        req->slotId = *slotId;
        req->continuation = true;
        req->prompt = std::string(req->deltaPrompt);  // Switch to delta prompt
        req->sessionId = std::to_string(*req->lookupHash);  // Set session ID to lookup hash as string
        info.validSessionFound = true;

        // Register under new hash for next turn
        sessionManager->registerPrefixHash(*slotId, req->registrationHash);

        onResolved(info);
        return;
      }
      // MISS: lookup hash not found, fall through to new session allocation
      TT_LOG_INFO(
          "[LLMController] Prefix cache MISS: hash={}, allocating new session",
          *req->lookupHash);
    } catch (const services::SessionInFlightException& e) {
      // All sessions with this hash are busy
      TT_LOG_WARN("[LLMController] All sessions busy for hash={}: {}",
                  *req->lookupHash, e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  // Legacy path: explicit sessionId (deprecated, sessionId is now a hash)
  // Try to parse sessionId as a hash for backward compatibility
  if (req->sessionId.has_value() && sessionManager && !req->hasPriorTurn) {
    try {
      // Try to parse sessionId as a numeric hash
      size_t hash = std::stoull(req->sessionId.value());
      auto slotId = sessionManager->acquireSessionSlot(hash);
      if (slotId != domain::INVALID_SLOT_ID) {
        req->slotId = slotId;
        req->continuation = true;
        info.validSessionFound = true;
        onResolved(info);
        return;
      } else {
        TT_LOG_INFO(
            "Received request with non existing session, resetting session id");
        req->sessionId.reset();
      }
    } catch (const std::exception& e) {
      // Failed to parse or session error - reset and fall through to new session
      TT_LOG_WARN("[LLMController] Legacy session lookup failed: {}, creating new session", e.what());
      req->sessionId.reset();
    }
  }

  // No prior turn OR lookup miss: allocate new session
  if (sessionManager) {
    sessionManager->createSession(
        [req, this, onResolved](const domain::Session& session) {
          req->sessionId = std::to_string(req->registrationHash);  // Use registration hash as session ID (string)
          req->slotId = session.getSlotId();
          req->continuation = false;

          // Register slot under registration hash for next turn's lookup
          if (sessionManager) {
            sessionManager->registerPrefixHash(session.getSlotId(), req->registrationHash);
            TT_LOG_INFO(
                "[LLMController] New session: slotId={}, registered under hash={}",
                session.getSlotId(), req->registrationHash);
          }

          SessionInfo info;
          onResolved(info);
        },
        [onError](std::string_view err) {
          onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
        },
        loop, "");  // Pass empty string for requestPrompt (no longer used for hash generation)
    return;
  }

  if (!sessionManager) {
    TT_LOG_WARN("[LLMController] SessionManager not available");
    onResolved(info);
  }
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
    auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
    auto cb =
        std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
            std::move(callback));

    resolveSession(
        request, loop,
        [this, request, cb](SessionInfo) {
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
                completion.usage.tps = completion.usage.completion_tokens *
                                       1000.0 / totalDuration.count();
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
              try {
                size_t hash = std::stoull(sessionId.value());
                sessionManager->setSessionInFlight(hash, false);
              } catch (const std::exception&) {
                // Ignore if sessionId is not a valid hash
              }
            }

            (*cb)(resp);
          } catch (const services::QueueFullException& e) {
            auto sessionId = request->sessionId;
            if (sessionId.has_value() && sessionManager) {
              try {
                size_t hash = std::stoull(sessionId.value());
                sessionManager->setSessionInFlight(hash, false);
              } catch (const std::exception&) {
                // Ignore if sessionId is not a valid hash
              }
            }
            (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                                "rate_limit_exceeded"));
          }
        },
        [cb](const SessionError& err) {
          TT_LOG_ERROR("[LLMController] Session resolution failed: {}",
                       err.message);
          if (err.type == SessionErrorType::RATE_LIMIT) {
            (*cb)(errorResponse(drogon::k429TooManyRequests, err.message,
                                "rate_limit_exceeded"));
          } else {
            (*cb)(errorResponse(
                drogon::k503ServiceUnavailable,
                std::string("Failed to allocate memory resources: ") +
                    err.message,
                "service_unavailable"));
          }
        });
  }
}

void LLMController::handleStreaming(
    std::shared_ptr<domain::LLMRequest> reqPtr,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::handleStreaming");

  auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
  auto cb =
      std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

  resolveSession(
      reqPtr, loop,
      [this, reqPtr, cb, loop](SessionInfo sessionInfo) {
        try {
          service->preProcess(*reqPtr);

          StreamParams params;
          params.completionId = "chatcmpl-" + std::to_string(reqPtr->task_id);
          params.model = reqPtr->model.value_or("default");
          params.created = static_cast<int64_t>(
              std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count());
          params.includeUsage = !reqPtr->stream_options.has_value() ||
                                reqPtr->stream_options->include_usage;
          params.continuousUsage =
              reqPtr->stream_options.has_value() &&
              reqPtr->stream_options->continuous_usage_stats;
          params.promptTokensCount = reqPtr->prompt_tokens_count;
          params.sessionId = reqPtr->sessionId;
          params.taskId = reqPtr->task_id;
          params.service = service;
          params.sessionManager = sessionManager;

          auto writer = SseStreamWriter::create(loop, std::move(params));

          auto streamingCallback = [writer](const domain::LLMStreamChunk& chunk,
                                            bool isFinal) {
            if (writer->isDone()) return;
            if (!chunk.choices.empty()) writer->handleTokenChunk(chunk);
            if (isFinal) writer->finalizeStream();
          };

          if (tt::config::llmMode() == tt::config::LLMMode::REGULAR) {
            // preprocess is already called
            service->submitStreamingRequest(*reqPtr, streamingCallback, true);
          } else if (tt::config::llmMode() ==
                     tt::config::LLMMode::DECODE_ONLY) {
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

          (*cb)(writer->buildResponse());
        } catch (const services::QueueFullException& e) {
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
        }
      },
      [cb](const SessionError& err) {
        TT_LOG_ERROR("[LLMController] Session resolution failed: {}",
                     err.message);
        if (err.type == SessionErrorType::RATE_LIMIT) {
          (*cb)(errorResponse(drogon::k429TooManyRequests, err.message,
                              "rate_limit_exceeded"));
        } else {
          (*cb)(errorResponse(
              drogon::k503ServiceUnavailable,
              std::string("Failed to allocate memory resources: ") +
                  err.message,
              "service_unavailable"));
        }
      });
}

bool LLMController::shouldDoPrefillOnDecode(const domain::LLMRequest& request,
                                            bool validSessionFound) const {
  if (validSessionFound) {
    return true;
  }

  const size_t maxTokens = tt::config::maxTokensToPrefillOnDecode();
  const size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  return promptTokens < maxTokens;
}

void LLMController::createSession(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  if (!sessionManager) {
    callback(errorResponse(drogon::k503ServiceUnavailable,
                           "Session management not available",
                           "service_unavailable"));
    return;
  }

  // Note: Pre-assigned slot_id is no longer supported in hash-based routing
  // Sessions are allocated dynamically by the session manager

  auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
  auto cb =
      std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

  sessionManager->createSession(
      [cb](const domain::Session& session) {
        Json::Value response = session.toJson();
        auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
        resp->setStatusCode(drogon::k201Created);
        (*cb)(resp);
      },
      [cb](std::string_view err) {
        (*cb)(errorResponse(drogon::k500InternalServerError, std::string(err),
                            "internal_error"));
      },
      loop, "");  // Empty requestPrompt
}

void LLMController::closeSession(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& sessionId) const {
  using tt::services::CloseSessionResult;

  // Parse sessionId as hash
  size_t hash;
  try {
    hash = std::stoull(sessionId);
  } catch (const std::exception&) {
    callback(errorResponse(drogon::k400BadRequest,
                          "Invalid session ID format (must be numeric hash)",
                          "invalid_session_id"));
    return;
  }

  auto result = sessionManager->closeSession(hash);

  switch (result) {
    case CloseSessionResult::SUCCESS: {
      Json::Value response;
      response["success"] = true;
      response["message"] = "Session closed";
      callback(drogon::HttpResponse::newHttpJsonResponse(response));
      break;
    }
    case CloseSessionResult::IN_FLIGHT:
      callback(errorResponse(drogon::k409Conflict,
                             "Session has an active request in flight; retry "
                             "after the request completes",
                             "session_in_flight"));
      break;
    case CloseSessionResult::NOT_FOUND:
      callback(errorResponse(drogon::k404NotFound, "Session not found",
                             "not_found"));
      break;
  }
}

void LLMController::getSlotId(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& sessionId) const {
  // Parse sessionId as hash
  size_t hash;
  try {
    hash = std::stoull(sessionId);
  } catch (const std::exception&) {
    callback(errorResponse(drogon::k400BadRequest,
                          "Invalid session ID format (must be numeric hash)",
                          "invalid_session_id"));
    return;
  }

  uint32_t slotId = sessionManager->getSlotIdBySessionId(hash);

  if (slotId == tt::domain::INVALID_SLOT_ID) {
    auto session = sessionManager->getSession(hash);
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
