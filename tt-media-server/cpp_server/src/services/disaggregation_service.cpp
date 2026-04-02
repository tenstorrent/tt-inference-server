// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/disaggregation_service.hpp"

#include "domain/completion_request.hpp"
#include "services/llm_service.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"

namespace tt::services {

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
        "[DisaggregationService] Health check from {} (active_tasks={})",
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

          auto response = domain::StreamingChunkResponse(message.task_id);
          response.choices.push_back(
              domain::CompletionChoice(message.generated_text));

          callback.value()(response, false);

          bool continueDecode = !message.token_ids.empty() &&
                                (!message.remaining_tokens.has_value() ||
                                 message.remaining_tokens.value() > 0);
          if (continueDecode) {
            auto request = domain::CompletionRequest(message.task_id);
            request.prompt = std::vector<int>(message.token_ids.begin(),
                                              message.token_ids.end());
            request.max_tokens = message.remaining_tokens;
            auto slotId = message.slot_id;
            request.slotId = slotId;
            llmService->submitStreamingRequest(request, callback.value());
          } else {
            auto finalResponse =
                domain::StreamingChunkResponse(message.task_id);
            domain::CompletionChoice finalChoice;
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
            auto response = domain::StreamingChunkResponse(taskId);
            response.choices.push_back(domain::CompletionChoice(""));
            response.choices.back().finish_reason = "error";
            callback(response, true);
          });
      streamCallbacks.clear();
    });
  }

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    socketService->onPrefillRequested(
        [this](const tt::sockets::PrefillRequestMessage& message) {
          auto request = domain::CompletionRequest(message.task_id);
          request.max_tokens = 1;
          auto maxTokens = message.max_tokens;
          using PromptVariant = std::variant<std::string, std::vector<int>>;

          request.prompt =
              message.token_ids.empty()
                  ? PromptVariant(message.prompt)
                  : PromptVariant(std::vector<int>(message.token_ids.begin(),
                                                   message.token_ids.end()));
          auto slotId = message.slot_id;

          llmService->submitStreamingRequest(
              request, [this, message, maxTokens, slotId](
                           const domain::StreamingChunkResponse& response,
                           bool /*isFinal*/) {
                auto remainingTokens =
                    maxTokens.has_value()
                        ? std::optional<int>(std::max(0, maxTokens.value() - 1))
                        : std::nullopt;

                auto prefillResult =
                    tt::sockets::PrefillResultMessage(message.task_id);
                prefillResult.remaining_tokens = remainingTokens;
                prefillResult.token_ids.insert(prefillResult.token_ids.end(),
                                               message.token_ids.begin(),
                                               message.token_ids.end());
                prefillResult.slot_id = slotId;
                if (response.choices.back().token_id.has_value()) {
                  prefillResult.token_ids.push_back(
                      response.choices.back().token_id.value());
                }
                prefillResult.generated_text = response.choices.back().text;
                socketService->sendPrefillResult(prefillResult);
              });
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
    domain::CompletionRequest& request, const StreamCallback& callback) {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    streamCallbacks.insert(request.task_id, callback);

    auto maxTokens = request.max_tokens;
    auto slotId = request.slotId;
    auto tokenIds = std::get<std::vector<int>>(request.prompt);
    auto sent = socketService->sendPrefillRequest(
        request.task_id, "",
        std::vector<int64_t>(tokenIds.begin(), tokenIds.end()), maxTokens,
        slotId);

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

}  // namespace tt::services
