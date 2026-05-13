// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/llm_controller.hpp"

#include <json/json.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "api/error_response.hpp"
#include "api/resolvers/chat_completions_resolver.hpp"
#include "api/resolvers/responses_resolver.hpp"
#include "api/resolvers/session_resolver.hpp"
#include "api/response_writer/non_stream_response_writer.hpp"
#include "api/response_writer/streaming_response_writer.hpp"
#include "api/stream_event_formatter.hpp"
#include "domain/llm/chat_completion_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/responses_request.hpp"
#include "domain/responses_response.hpp"
#include "profiling/tracy.hpp"
#include "services/service_container.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"

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
  chatResolver = c.chatCompletionsResolver();
  respResolver = c.responsesResolver();

  if (!service) {
    throw std::runtime_error(
        "[LLMController] LLM service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  if (!chatResolver || !respResolver) {
    throw std::runtime_error(
        "[LLMController] Session resolvers not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[LLMController] Initialized (service already started)");
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

  auto reqPtr = std::make_shared<LLMRequest>(chatReq.toLLMRequest());
  const size_t maxContextLength = tt::config::maxContextLength();
  const size_t promptTokens =
      static_cast<size_t>(std::max(0, reqPtr->full_prompt_tokens_count));

  const size_t requested =
      promptTokens +
      (reqPtr->max_tokens.has_value()
           ? static_cast<size_t>(std::max(0, *reqPtr->max_tokens))
           : 1);
  const bool exceedsContext = requested > maxContextLength;

  if (exceedsContext) {
    std::string detail =
        "prompt_tokens=" + std::to_string(reqPtr->full_prompt_tokens_count);
    if (reqPtr->max_tokens.has_value()) {
      detail += ", max_tokens=" + std::to_string(*reqPtr->max_tokens);
    }
    callback(errorResponse(drogon::k400BadRequest,
                           "Request exceeds maximum context length (" +
                               std::to_string(maxContextLength) +
                               " tokens): " + detail,
                           "invalid_request_error"));
    return;
  }

  if (reqPtr->stream) {
    const bool includeUsage = !reqPtr->stream_options.has_value() ||
                              reqPtr->stream_options->include_usage;
    handleStreaming(reqPtr, std::make_shared<ChatCompletionEventFormatter>(),
                    includeUsage, *chatResolver, std::move(callback));
  } else {
    handleNonStreaming(reqPtr, /*builder=*/nullptr, *chatResolver,
                       std::move(callback));
  }
}

void LLMController::responses(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::responses");

  auto json = req->getJsonObject();
  if (!json) {
    callback(errorResponse(drogon::k400BadRequest, "Invalid JSON body",
                           "invalid_request_error"));
    return;
  }

  std::optional<domain::ResponsesRequest> respReqOpt;
  try {
    uint32_t taskId = tt::utils::TaskIDGenerator::generate();
    respReqOpt = domain::ResponsesRequest::fromJson(*json, std::move(taskId));
  } catch (const std::exception& e) {
    callback(errorResponse(drogon::k400BadRequest,
                           std::string("Failed to parse request: ") + e.what(),
                           "invalid_request_error"));
    return;
  }

  auto respReqPtr =
      std::make_shared<domain::ResponsesRequest>(std::move(*respReqOpt));
  const domain::ResponsesRequest& respReq = *respReqPtr;

  TT_LOG_INFO("[LLMController] /v1/responses task_id={} model={}",
              respReq.task_id, respReq.model.value_or("default"));

  if (!service->isModelReady()) {
    callback(errorResponse(drogon::k503ServiceUnavailable, "Model is not ready",
                           "service_unavailable"));
    return;
  }

  auto reqPtr = std::make_shared<LLMRequest>(respReq.toLLMRequest());
  auto samplingParams = tt::utils::mapper::mapSamplingParams(*reqPtr);

  if (reqPtr->stream) {
    auto formatter =
        std::make_shared<ResponsesEventFormatter>(respReqPtr, samplingParams);
    handleStreaming(reqPtr, std::move(formatter), /*includeUsage=*/true,
                    *respResolver, std::move(callback));
    return;
  }

  auto builder = [respReqPtr, samplingParams](
                     const LLMResponse& completion) -> std::string {
    int64_t createdAt = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    Json::Value output(Json::arrayValue);
    for (const auto& choice : completion.choices) {
      Json::Value item;
      item["type"] = "message";
      item["id"] = "msg_" + std::to_string(choice.index);
      item["status"] = "completed";
      item["role"] = "assistant";

      Json::Value content(Json::arrayValue);
      Json::Value textPart;
      textPart["type"] = "output_text";
      textPart["text"] = choice.text;
      content.append(std::move(textPart));

      if (choice.reasoning.has_value()) {
        Json::Value reasoningPart;
        reasoningPart["type"] = "reasoning";
        reasoningPart["text"] = *choice.reasoning;
        content.append(std::move(reasoningPart));
      }

      item["content"] = std::move(content);
      output.append(std::move(item));
    }

    domain::ResponseUsage usage;
    usage.input_tokens = completion.usage.prompt_tokens;
    usage.output_tokens = completion.usage.completion_tokens;
    usage.total_tokens = completion.usage.total_tokens;

    std::string status =
        (!completion.choices.empty() &&
         completion.choices[0].finish_reason.value_or("stop") == "length")
            ? "incomplete"
            : "completed";

    auto resp = domain::ResponsesResponse::fromRequest(
        completion.task_id, *respReqPtr, samplingParams, completion.model,
        createdAt, std::move(output), std::move(status), std::move(usage));

    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    writer["emitUTF8"] = true;
    return Json::writeString(writer, resp.toOpenaiJson());
  };

  handleNonStreaming(reqPtr, std::move(builder), *respResolver,
                     std::move(callback));
}

ResponseWriterParams LLMController::makeWriterParams(
    const LLMRequest& request, services::SlotLease lease) const {
  ResponseWriterParams params;
  params.completionId = "chatcmpl-" + std::to_string(request.task_id);
  params.model = request.model.value_or("default");
  params.created = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  params.promptTokenCount = request.full_prompt_tokens_count > 0
                                ? request.full_prompt_tokens_count
                                : request.prompt_tokens_count;
  params.cachedTokenCount =
      request.continuation
          ? request.full_prompt_tokens_count - request.prompt_tokens_count
          : 0;
  params.sessionId = request.sessionId;
  params.taskId = request.task_id;
  params.service = service;
  params.lease = std::move(lease);
  return params;
}

std::function<void(const LLMStreamChunk&, bool)>
LLMController::makeStreamingCallback(std::shared_ptr<ResponseWriter> writer) {
  return [writer = std::move(writer)](const LLMStreamChunk& chunk,
                                      bool isFinal) {
    if (writer->isDone()) return;
    if (!chunk.choices.empty()) writer->handleTokenChunk(chunk);
    // finalize() releases the SlotLease held in the writer's params,
    // so the session becomes idle the moment the generator is done.
    if (isFinal) writer->finalize();
  };
}

drogon::HttpResponsePtr LLMController::makeSessionErrorResponse(
    const resolvers::SessionError& err) {
  if (err.type == resolvers::SessionErrorType::RATE_LIMIT) {
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
    std::shared_ptr<StreamEventFormatter> formatter, bool includeUsage,
    const resolvers::SessionResolver& resolver,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::handleStreaming");

  auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
  auto cb =
      std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

  auto cancelFn = [svc = service, taskId = reqPtr->task_id]() {
    svc->abortRequest(taskId);
  };

  resolver.resolve(
      reqPtr, loop,
      [this, reqPtr, cb, loop, formatter = std::move(formatter),
       includeUsage](services::SlotLease lease) mutable {
        // The lease lives in this lambda until it is moved into the
        // writer's params. Any early-return below destroys it, which
        // releases the session's in-flight grant.
        try {
          service->preProcess(*reqPtr);
        } catch (const services::QueueFullException& e) {
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          (*cb)(errorResponse(drogon::k400BadRequest, e.what(),
                              "invalid_request_error"));
          return;
        }

        auto writer = StreamingResponseWriter::create(
            loop, makeWriterParams(*reqPtr, std::move(lease)), includeUsage,
            formatter);

        try {
          dispatchGeneration(*reqPtr, makeStreamingCallback(writer));
        } catch (const services::QueueFullException& e) {
          writer->abort();
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          writer->abort();
          (*cb)(errorResponse(drogon::k500InternalServerError, e.what(),
                              "internal_error"));
          return;
        }

        (*cb)(writer->buildResponse());
      },
      [cb](const resolvers::SessionError& err) {
        TT_LOG_ERROR("[LLMController] Session resolution failed: {}",
                     err.message);
        (*cb)(makeSessionErrorResponse(err));
      },
      std::move(cancelFn));
}

void LLMController::handleNonStreaming(
    std::shared_ptr<LLMRequest> reqPtr,
    NonStreamResponseWriter::ResponseBuilder builder,
    const resolvers::SessionResolver& resolver,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  ZoneScopedN("API::handleNonStreaming");

  auto* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
  auto cb =
      std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

  auto cancelFn = [svc = service, taskId = reqPtr->task_id]() {
    svc->abortRequest(taskId);
  };

  resolver.resolve(
      reqPtr, loop,
      [this, reqPtr, cb, builder = std::move(builder)](
          services::SlotLease lease) mutable {
        try {
          service->preProcess(*reqPtr);
        } catch (const services::QueueFullException& e) {
          (*cb)(errorResponse(drogon::k429TooManyRequests, e.what(),
                              "rate_limit_exceeded"));
          return;
        } catch (const std::exception& e) {
          (*cb)(errorResponse(drogon::k400BadRequest, e.what(),
                              "invalid_request_error"));
          return;
        }

        // Move the http callback into the writer; from here on out every
        // success/error path goes through writer->finalize / sendError so
        // the response is delivered exactly once and the SlotLease the
        // writer holds is released exactly once.
        auto writer = NonStreamResponseWriter::create(
            makeWriterParams(*reqPtr, std::move(lease)), std::move(*cb),
            std::move(builder));

        try {
          dispatchGeneration(*reqPtr, makeStreamingCallback(writer));
        } catch (const services::QueueFullException& e) {
          writer->sendError(drogon::k429TooManyRequests, e.what(),
                            "rate_limit_exceeded");
        } catch (const std::exception& e) {
          writer->sendError(drogon::k500InternalServerError, e.what(),
                            "internal_error");
        }
      },
      [cb](const resolvers::SessionError& err) {
        TT_LOG_ERROR("[LLMController] Session resolution failed: {}",
                     err.message);
        (*cb)(makeSessionErrorResponse(err));
      },
      std::move(cancelFn));
}

void LLMController::dispatchGeneration(
    LLMRequest& request,
    const std::function<void(const LLMStreamChunk&, bool)>& cb) const {
  const auto mode = tt::config::llmMode();
  if (mode == tt::config::LLMMode::REGULAR) {
    service->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    return;
  }

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    if (shouldDoPrefillOnDecode(request,
                                /*validSessionFound=*/request.continuation)) {
      TT_LOG_DEBUG("[LLMController] Using prefill on decode for sessionId: {}",
                   request.sessionId.value_or("none"));
      service->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    } else {
      TT_LOG_DEBUG(
          "[LLMController] Using disaggregated prefill for request with "
          "sessionId: {}",
          request.sessionId.value_or("none"));
      disaggregationService->handleStreamingRequest(
          request, request.registrationHash, cb);
    }
    return;
  }

  throw std::runtime_error(
      "LLM Mode must be regular or decode only for chat completions");
}

bool LLMController::shouldDoPrefillOnDecode(const LLMRequest& request,
                                            bool validSessionFound) const {
  const bool socketReady = socketService && socketService->isConnected();
  if (!socketReady) {
    TT_LOG_WARN(
        "[LLMController] Prefill server not connected; falling back to "
        "prefill on decode for taskId={}",
        request.task_id);
    return true;
  }

  if (request.disaggregation_override.has_value()) {
    const bool forceDisagg = *request.disaggregation_override;
    TT_LOG_INFO(
        "[LLMController] Honoring disaggregation override={} for taskId={}",
        forceDisagg, request.task_id);
    return !forceDisagg;
  }

  if (validSessionFound) {
    return true;
  }

  const size_t maxTokens = tt::config::maxTokensToPrefillOnDecode();
  const size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  return promptTokens < maxTokens;
}

}  // namespace tt::api
