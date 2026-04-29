// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/llm_controller.hpp"

#include <json/json.h>

#include <chrono>
#include <functional>
#include <memory>
#include <optional>

#include "api/error_response.hpp"
#include "api/response_writer/non_stream_response_writer.hpp"
#include "api/response_writer/streaming_response_writer.hpp"
#include "config/settings.hpp"
#include "domain/chat_completion_request.hpp"
#include "domain/models_response.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "services/service_container.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

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

  const auto& c = tt::services::ServiceContainer::instance();
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
    std::function<void(const SessionError&)> onError,
    std::function<void()> cancelFn) const {
  SessionInfo info;

  if (!sessionManager) {
    TT_LOG_WARN("[LLMController] SessionManager not available");
    onResolved(info);
    return;
  }

  // Routing information derived once from the request's chat messages.
  auto routingInfo = tt::utils::computePrefixCachingInfo(req->messages);
  TT_LOG_DEBUG(
      "[LLMController] Routing: hasPriorTurn={}, lookupHash={}, "
      "registrationHash={}",
      routingInfo.hasPriorTurn,
      routingInfo.lookupHash.has_value()
          ? std::to_string(*routingInfo.lookupHash)
          : "none",
      routingInfo.registrationHash);

  // Layer 1: Legacy path — client-provided sessionId wins if it resolves.
  if (req->sessionId.has_value()) {
    const std::string sessionId = req->sessionId.value();
    try {
      auto slotId = sessionManager->acquireInFlight(sessionId, cancelFn);
      if (slotId != domain::INVALID_SLOT_ID) {
        req->slotId = slotId;
        req->continuation = true;
        req->prompt = routingInfo.deltaPrompt;
        sessionManager->registerPrefixHash(sessionId,
                                           routingInfo.registrationHash);
        TT_LOG_INFO(
            "[LLMController] Legacy session continue: sessionId={}, "
            "slotId={}, registered under hash={}",
            sessionId, slotId, routingInfo.registrationHash);
        info.validSessionFound = true;
        onResolved(info);
        return;
      }

      TT_LOG_INFO(
          "[LLMController] sessionId={} not found; falling back to prefix "
          "cache",
          sessionId);
      req->sessionId.reset();
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[LLMController] Session {} is busy: {}", sessionId,
                  e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    } catch (const std::exception& e) {
      TT_LOG_WARN(
          "[LLMController] Legacy session acquisition failed for {}: {}; "
          "falling back to prefix cache",
          sessionId, e.what());
      req->sessionId.reset();
    }
  }

  // Layer 2: Prefix-cache routing. Requires a prior [assistant, user] pair
  if (routingInfo.hasPriorTurn && routingInfo.lookupHash.has_value()) {
    try {
      auto acquired = sessionManager->tryAcquireByPrefixHash(
          *routingInfo.lookupHash, cancelFn);

      if (acquired.has_value()) {
        // HIT: found matching session, send delta only
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        TT_LOG_DEBUG(
            "[LLMController] Prefix cache HIT: hash={}, sessionId={}, "
            "slotId={}",
            *routingInfo.lookupHash, acquired->sessionId, acquired->slotId);
        req->slotId = acquired->slotId;
        req->sessionId = acquired->sessionId;
        req->continuation = true;
        req->prompt = routingInfo.deltaPrompt;
        sessionManager->registerPrefixHash(acquired->sessionId,
                                           routingInfo.registrationHash);
        info.validSessionFound = true;
        onResolved(info);
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_DEBUG(
          "[LLMController] Prefix cache MISS: hash={}, allocating new session",
          *routingInfo.lookupHash);
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[LLMController] All sessions busy for hash={}: {}",
                  *routingInfo.lookupHash, e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  // Layer 3: Allocate a new session. Async — onCompletion runs on loop.
  sessionManager->createSession(
      [req, routingInfo, onResolved, cancelFn = std::move(cancelFn),
       mgr = sessionManager](const domain::Session& session) mutable {
        req->sessionId = session.getSessionId();
        req->slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        req->continuation = false;
        mgr->registerPrefixHash(session.getSessionId(),
                                routingInfo.registrationHash);
        TT_LOG_INFO(
            "[LLMController] New session: sessionId={}, slotId={}, "
            "registered under hash={}",
            session.getSessionId(),
            req->slotId.has_value() ? std::to_string(*req->slotId) : "none",
            routingInfo.registrationHash);

        SessionInfo info;
        onResolved(info);
      },
      [onError](std::string_view err) {
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, routingInfo.registrationHash);
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
    handleNonStreaming(request, std::move(callback));
  }
}

ResponseWriterParams LLMController::makeWriterParams(
    const domain::LLMRequest& request) const {
  ResponseWriterParams params;
  params.completionId = "chatcmpl-" + std::to_string(request.task_id);
  params.model = request.model.value_or("default");
  params.created = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  params.promptTokenCount = request.prompt_tokens_count;
  params.sessionId = request.sessionId;
  params.taskId = request.task_id;
  params.service = service;
  params.sessionManager = sessionManager;
  return params;
}

std::function<void(const domain::LLMStreamChunk&, bool)>
LLMController::makeStreamingCallback(std::shared_ptr<ResponseWriter> writer) {
  return [writer = std::move(writer)](const domain::LLMStreamChunk& chunk,
                                      bool isFinal) {
    if (writer->isDone()) return;
    if (!chunk.choices.empty()) writer->handleTokenChunk(chunk);
    if (isFinal) writer->finalize();
  };
}

drogon::HttpResponsePtr LLMController::makeSessionErrorResponse(
    const SessionError& err) {
  if (err.type == SessionErrorType::RATE_LIMIT) {
    return errorResponse(drogon::k429TooManyRequests, err.message,
                         "rate_limit_exceeded");
  }
  return errorResponse(
      drogon::k503ServiceUnavailable,
      std::string("Failed to allocate memory resources: ") + err.message,
      "service_unavailable");
}

void LLMController::handleStreaming(
    std::shared_ptr<domain::LLMRequest> reqPtr,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::handleStreaming");

  auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
  auto cb =
      std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

  auto cancelFn = [svc = service, taskId = reqPtr->task_id]() {
    svc->abortRequest(taskId);
  };

  resolveSession(
      reqPtr, loop,
      [this, reqPtr, cb, loop](SessionInfo sessionInfo) {
        try {
          service->preProcess(*reqPtr);
        } catch (const services::QueueFullException& e) {
          releaseSessionInFlight(reqPtr->sessionId);
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          releaseSessionInFlight(reqPtr->sessionId);
          (*cb)(errorResponse(drogon::k400BadRequest, e.what(),
                              "invalid_request_error"));
          return;
        }

        const bool includeUsage = !reqPtr->stream_options.has_value() ||
                                  reqPtr->stream_options->include_usage;
        const bool continuousUsage =
            reqPtr->stream_options.has_value() &&
            reqPtr->stream_options->continuous_usage_stats;

        auto writer = StreamingResponseWriter::create(
            loop, makeWriterParams(*reqPtr), includeUsage, continuousUsage);

        try {
          dispatchGeneration(*reqPtr, sessionInfo.validSessionFound,
                             makeStreamingCallback(writer));
        } catch (const services::QueueFullException& e) {
          releaseSessionInFlight(reqPtr->sessionId);
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          releaseSessionInFlight(reqPtr->sessionId);
          (*cb)(errorResponse(drogon::k500InternalServerError, e.what(),
                              "internal_error"));
          return;
        }

        (*cb)(writer->buildResponse());
      },
      [cb](const SessionError& err) {
        TT_LOG_ERROR("[LLMController] Session resolution failed: {}",
                     err.message);
        (*cb)(makeSessionErrorResponse(err));
      },
      std::move(cancelFn));
}

void LLMController::handleNonStreaming(
    std::shared_ptr<domain::LLMRequest> reqPtr,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::handleNonStreaming");

  auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
  auto cb =
      std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

  auto cancelFn = [svc = service, taskId = reqPtr->task_id]() {
    svc->abortRequest(taskId);
  };

  resolveSession(
      reqPtr, loop,
      [this, reqPtr, cb](SessionInfo sessionInfo) {
        try {
          service->preProcess(*reqPtr);
        } catch (const services::QueueFullException& e) {
          releaseSessionInFlight(reqPtr->sessionId);
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          releaseSessionInFlight(reqPtr->sessionId);
          (*cb)(errorResponse(drogon::k400BadRequest, e.what(),
                              "invalid_request_error"));
          return;
        }

        // Move the http callback into the writer; from here on out every
        // success/error path goes through writer->finalize / sendError so
        // the response is delivered exactly once and the session in-flight
        // slot is always released.
        auto writer = NonStreamResponseWriter::create(makeWriterParams(*reqPtr),
                                                      std::move(*cb));

        try {
          dispatchGeneration(*reqPtr, sessionInfo.validSessionFound,
                             makeStreamingCallback(writer));
        } catch (const services::QueueFullException& e) {
          writer->sendError(drogon::k429TooManyRequests, e.what(),
                            "rate_limit_exceeded");
        } catch (const std::exception& e) {
          writer->sendError(drogon::k500InternalServerError, e.what(),
                            "internal_error");
        }
      },
      [cb](const SessionError& err) {
        TT_LOG_ERROR("[LLMController] Session resolution failed: {}",
                     err.message);
        (*cb)(makeSessionErrorResponse(err));
      },
      std::move(cancelFn));
}

void LLMController::dispatchGeneration(
    domain::LLMRequest& request, bool validSessionFound,
    const std::function<void(const domain::LLMStreamChunk&, bool)>& cb) const {
  const auto mode = tt::config::llmMode();
  if (mode == tt::config::LLMMode::REGULAR) {
    service->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    return;
  }

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    if (shouldDoPrefillOnDecode(request, validSessionFound)) {
      TT_LOG_DEBUG("[LLMController] Using prefill on decode for sessionId: {}",
                   request.sessionId.value_or("none"));
      service->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    } else {
      TT_LOG_DEBUG(
          "[LLMController] Using disaggregated prefill for request with "
          "sessionId: {}",
          request.sessionId.value_or("none"));
      disaggregationService->handleStreamingRequest(request, cb);
    }
    return;
  }

  throw std::runtime_error(
      "LLM Mode must be regular or decode only for chat completions");
}

void LLMController::releaseSessionInFlight(
    const std::optional<std::string>& sessionId) const {
  if (sessionId.has_value() && sessionManager) {
    sessionManager->releaseInFlight(sessionId.value());
  }
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

  // Parse optional slot_id from request body
  std::optional<uint32_t> slotId;
  auto json = req->getJsonObject();
  if (json && json->isMember("slot_id")) {
    slotId = (*json)["slot_id"].asUInt();
  }
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
      loop, 0, slotId);
}

void LLMController::closeSession(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& sessionId) const {
  using tt::services::CloseSessionResult;

  auto result = sessionManager->closeSession(sessionId);

  switch (result) {
    case CloseSessionResult::SUCCESS: {
      Json::Value response;
      response["success"] = true;
      response["message"] = "Session closed";
      callback(drogon::HttpResponse::newHttpJsonResponse(response));
      break;
    }
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
  uint32_t slotId = sessionManager->getSlotIdBySessionId(sessionId);

  if (slotId == tt::domain::INVALID_SLOT_ID) {
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
