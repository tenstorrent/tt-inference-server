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
#include "domain/llm/chat_completion_request.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "services/service_container.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::api {

LLMController::LLMController() {
  if (!tt::config::isLlmServiceEnabled()) {
    TT_LOG_INFO(
        "[LLMController] Skipping initialization (TT_model_SERVICE != llm)");
    return;
  }

  tt::config::model();

  const auto& c = tt::services::ServiceContainer::instance();
  service = std::dynamic_pointer_cast<tt::services::LLMService>(
      c.getService(tt::config::ModelService::LLM));
  disaggregationService = c.disaggregation();
  sessionManager = c.sessionManager();
  socketService = c.socket();

  if (!service) {
    throw std::runtime_error(
        "[LLMController] LLM service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[LLMController] Initialized (service already started)");
}

void LLMController::resolveSession(
    std::shared_ptr<LLMRequest> req, trantor::EventLoop* loop,
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

  // Layer 1: Prefix-cache routing. Requires a prior [assistant, user] pair
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
        req->session = sessionManager->getSession(acquired->sessionId);
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

  // Layer 2: Allocate a new session. Async — onCompletion runs on loop.
  sessionManager->createSession(
      [req, routingInfo, onResolved, cancelFn = std::move(cancelFn),
       mgr = sessionManager](const domain::Session& session) mutable {
        req->sessionId = session.getSessionId();
        req->slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        req->session = mgr->getSession(session.getSessionId());
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

  std::optional<ChatCompletionRequest> chatReqOpt;
  try {
    uint32_t taskId = tt::utils::TaskIDGenerator::generate();
    chatReqOpt = ChatCompletionRequest::fromJson(*json, std::move(taskId));
  } catch (const std::exception& e) {
    callback(errorResponse(drogon::k400BadRequest,
                           std::string("Failed to parse request: ") + e.what(),
                           "invalid_request_error"));
    return;
  }

  ChatCompletionRequest& chatReq = *chatReqOpt;

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

  auto request = std::make_shared<LLMRequest>(chatReq.toLLMRequest());

  if (request->stream) {
    handleStreaming(request, std::move(callback));
  } else {
    handleNonStreaming(request, std::move(callback));
  }
}

ResponseWriterParams LLMController::makeWriterParams(
    const LLMRequest& request) const {
  ResponseWriterParams params;
  params.completionId = "chatcmpl-" + std::to_string(request.task_id);
  params.model = request.model.value_or("default");
  params.created = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  params.promptTokenCount = request.prompt_tokens_count;
  params.taskId = request.task_id;
  params.service = service;
  if (request.session) {
    params.onSessionRelease = [s = request.session]() { s->clearInFlight(); };
  }
  return params;
}

std::function<void(const LLMStreamChunk&, bool)>
LLMController::makeStreamingCallback(std::shared_ptr<ResponseWriter> writer,
                                     domain::Session* session) {
  return [writer = std::move(writer), session](const LLMStreamChunk& chunk,
                                               bool isFinal) {
    if (writer->isDone()) return;
    if (!chunk.choices.empty()) writer->handleTokenChunk(chunk);
    if (isFinal) {
      if (session) session->clearInFlight();
      writer->finalize();
    }
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
    std::shared_ptr<LLMRequest> reqPtr,
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
          if (reqPtr->session) reqPtr->session->clearInFlight();
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          if (reqPtr->session) reqPtr->session->clearInFlight();
          (*cb)(errorResponse(drogon::k400BadRequest, e.what(),
                              "invalid_request_error"));
          return;
        }

        const bool includeUsage = !reqPtr->stream_options.has_value() ||
                                  reqPtr->stream_options->include_usage;

        auto writer = StreamingResponseWriter::create(
            loop, makeWriterParams(*reqPtr), includeUsage);

        try {
          dispatchGeneration(*reqPtr, sessionInfo.validSessionFound,
                             makeStreamingCallback(writer, reqPtr->session));
        } catch (const services::QueueFullException& e) {
          if (reqPtr->session) reqPtr->session->clearInFlight();
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          if (reqPtr->session) reqPtr->session->clearInFlight();
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
    std::shared_ptr<LLMRequest> reqPtr,
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
          if (reqPtr->session) reqPtr->session->clearInFlight();
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          if (reqPtr->session) reqPtr->session->clearInFlight();
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
                             makeStreamingCallback(writer, reqPtr->session));
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
    LLMRequest& request, bool validSessionFound,
    const std::function<void(const LLMStreamChunk&, bool)>& cb) const {
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

bool LLMController::shouldDoPrefillOnDecode(const LLMRequest& request,
                                            bool validSessionFound) const {
  if (validSessionFound) {
    return true;
  }

  // In disaggregated decode mode, fall back to running prefill locally if the
  // prefill server socket is unavailable — otherwise the request would be sent
  // to a peer that cannot service it.
  if (!socketService || !socketService->isConnected()) {
    TT_LOG_WARN(
        "[LLMController] Prefill server not connected; falling back to "
        "prefill on decode for taskId={}",
        request.task_id);
    return true;
  }

  const size_t maxTokens = tt::config::maxTokensToPrefillOnDecode();
  const size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  return promptTokens < maxTokens;
}

}  // namespace tt::api
