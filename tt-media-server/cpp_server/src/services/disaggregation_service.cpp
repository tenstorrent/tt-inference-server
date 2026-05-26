// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/disaggregation_service.hpp"

#include "domain/llm/llm_request.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/llm_service.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"

namespace tt::services {

using namespace tt::domain::llm;

DisaggregationService::DisaggregationService(
    tt::config::LLMMode mode, std::shared_ptr<LLMService> llmService,
    std::shared_ptr<sockets::InterServerService> socketService)
    : mode(mode),
      llmService(std::move(llmService)),
      socketService(std::move(socketService)) {
  setupSocketHandlers();
}

void DisaggregationService::setupSocketHandlers() {
  socketService->setHealthCheckCallback([](const std::string& serverId,
                                           double /*cpu*/, double /*memory*/,
                                           int tasks) {
    TT_LOG_INFO(
        "[DisaggregationService]  Health check from {} (active_tasks={})",
        serverId, tasks);
  });

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    socketService->onPrefillComplete(
        [this](const tt::sockets::PrefillResultMessage& message) {
          auto callback = streamCallbacks.get(message.task_id);
          if (!callback.has_value()) {
            TT_LOG_WARN("[DisaggregationService] No callback for task_id: {}",
                        message.task_id);
            return;
          }
          streamCallbacks.erase(message.task_id);

          if (message.error) {
            TT_LOG_ERROR(
                "[DisaggregationService] Prefill error received for task {}, "
                "propagating error to client",
                message.task_id);
            callback.value()(makeErrorChunk(message.task_id, "prefill error"),
                             /*isFinal=*/true);
            return;
          }

          auto response = LLMStreamChunk(message.task_id);
          LLMChoice choice;
          choice.text = message.generated_text;
          response.choices.push_back(std::move(choice));

          callback.value()(response, false);

          bool continueDecode = !message.token_ids.empty() &&
                                (!message.remaining_tokens.has_value() ||
                                 message.remaining_tokens.value() > 0);
          if (continueDecode) {
            if (auto* reasoningParser = llmService->getReasoningParser()) {
              reasoningParser->initializeTask(message.task_id);
              reasoningParser->processToken(message.task_id,
                                            message.token_ids.back(),
                                            /*decodedText=*/"");
            }
            auto request = LLMRequest(message.task_id);
            request.disaggregated = true;
            request.prompt.emplace<std::vector<int>>(message.token_ids.begin(),
                                                     message.token_ids.end());
            request.max_tokens = message.remaining_tokens;
            request.slotId = message.slot_id;
            // Restore the sampling subset echoed back from the prefill server.
            request.temperature = message.temperature;
            request.top_p = message.top_p;
            request.top_k = message.top_k;
            request.fast_mode = message.fast_mode;
            llmService->submitStreamingRequest(request, callback.value());
          } else {
            auto finalResponse = LLMStreamChunk(message.task_id);
            LLMChoice finalChoice;
            finalChoice.text = "";
            finalChoice.index = 0;
            finalChoice.finish_reason = "stop";
            finalResponse.choices.push_back(std::move(finalChoice));
            callback.value()(finalResponse, true);
          }
        });

    socketService->setConnectionLostCallback([this]() {
      streamCallbacks.forEach(
          [](uint32_t taskId, const StreamCallback& callback) {
            callback(makeErrorChunk(taskId, "connection lost"),
                     /*isFinal=*/true);
          });
      streamCallbacks.clear();
    });
  }

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    // On prefill runner death, drop the inter-server socket so the decode
    // side's connection-lost handler can fail in-flight streams instead of
    // hanging on requests nobody can answer.  Assumes WorkerManager does not
    // auto-restart workers; if that changes, the callback will fire again on
    // the replacement worker's first crash and stop a possibly-rearmed socket.
    if (auto* workerManager = llmService->getWorkerManager()) {
      workerManager->setWorkerDeathCallback(
          [socket = socketService](size_t workerIdx, pid_t pid) {
            TT_LOG_ERROR(
                "[DisaggregationService] Prefill runner (worker {}, PID {}) "
                "is down; disconnecting inter-server socket",
                workerIdx, pid);
            socket->stop();
          });
    }

    socketService->onPrefillRequested(
        [this](const tt::sockets::PrefillRequestMessage& message) {
          auto request = LLMRequest(message.task_id);
          request.max_tokens = 1;
          request.temperature = message.temperature;
          request.top_p = message.top_p;
          request.top_k = message.top_k;
          request.fast_mode = message.fast_mode;

          auto maxTokens = message.max_tokens;

          request.prompt.emplace<std::vector<int>>(message.token_ids.begin(),
                                                   message.token_ids.end());
          auto slotId = message.slot_id;
          request.slotId = slotId;

          llmService->submitStreamingRequest(
              request, [this, message, maxTokens, slotId](
                           const LLMStreamChunk& response, bool /*isFinal*/) {
                auto prefillResult =
                    tt::sockets::PrefillResultMessage(message.task_id);
                prefillResult.slot_id = slotId;
                prefillResult.temperature = message.temperature;
                prefillResult.top_p = message.top_p;
                prefillResult.top_k = message.top_k;
                prefillResult.fast_mode = message.fast_mode;

                bool isError = !response.choices.empty() &&
                               response.choices.back().finish_reason == "error";
                if (isError) {
                  TT_LOG_WARN(
                      "[DisaggregationService] Prefill error for task {}, "
                      "propagating to decode server",
                      message.task_id);
                  prefillResult.error = true;
                  prefillResult.finished = true;
                } else {
                  prefillResult.remaining_tokens =
                      maxTokens.has_value() ? std::optional<int>(std::max(
                                                  0, maxTokens.value() - 1))
                                            : std::nullopt;
                  prefillResult.token_ids.insert(prefillResult.token_ids.end(),
                                                 message.token_ids.begin(),
                                                 message.token_ids.end());
                  if (response.choices.back().token_id.has_value()) {
                    prefillResult.token_ids.push_back(
                        response.choices.back().token_id.value());
                  }
                  prefillResult.generated_text = response.choices.back().text;
                }

                socketService->sendPrefillResult(prefillResult);
              });
        });

    socketService->onPrefillCancelled(
        [this](const tt::sockets::CancelPrefillMessage& message) {
          llmService->abortRequest(message.task_id);
        });
  }
}

DisaggregationService::~DisaggregationService() { stop(); }

void DisaggregationService::start() {
  if (socketService->isEnabled()) {
    socketService->start();
  }
}

void DisaggregationService::stop() { socketService->stop(); }

void DisaggregationService::handleStreamingRequest(
    LLMRequest& request, size_t requestHash, const StreamCallback& callback) {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    streamCallbacks.insert(request.task_id, callback);

    auto maxTokens = request.max_tokens;
    auto slotId = request.slotId;
    auto tokenIds = std::get<std::vector<int>>(request.prompt);
    auto sent = socketService->sendPrefillRequest(
        request.task_id, requestHash,
        std::vector<int64_t>(tokenIds.begin(), tokenIds.end()), maxTokens,
        slotId, tt::utils::mapper::mapSamplingParams(request));

    if (!sent) {
      streamCallbacks.erase(request.task_id);
      throw std::runtime_error(
          "[DisaggregationService] Failed to send prefill request for "
          "task_id: " +
          std::to_string(request.task_id));
    }
  } else {
    throw std::runtime_error(
        "[DisaggregationService] Server must be in decode only mode to handle "
        "streaming requests");
  }
}

void DisaggregationService::abortRequest(uint32_t taskId) {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    auto callback = streamCallbacks.take(taskId);
    if (!callback.has_value()) {
      return;
    }

    bool sent = socketService->sendPrefillCancel(taskId);
    if (!sent) {
      TT_LOG_WARN(
          "[DisaggregationService] Failed to send prefill cancel for task_id: "
          "{}",
          taskId);
    }

    LLMStreamChunk abortResponse{taskId};
    LLMChoice choice;
    choice.finish_reason = "abort";
    abortResponse.choices.push_back(std::move(choice));
    callback.value()(abortResponse, /*isFinal=*/true);
    return;
  }

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    llmService->abortRequest(taskId);
  }
}

}  // namespace tt::services
