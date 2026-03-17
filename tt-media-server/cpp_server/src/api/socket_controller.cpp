// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/socket_controller.hpp"

#include <memory>

#include "config/settings.hpp"
#include "domain/completion_request.hpp"
#include "services/llm_service.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"

namespace tt::api {

SocketController::SocketController(
    std::shared_ptr<services::LLMService> llmService,
    std::shared_ptr<sockets::InterServerService> socketService)
    : llm_service_(std::move(llmService)),
      socket_service_(std::move(socketService)) {
  if (!socket_service_ || !socket_service_->isEnabled()) {
    TT_LOG_INFO(
        "[SocketController] Socket service not enabled, skipping handler "
        "setup");
    return;
  }

  auto mode = tt::config::llmMode();

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    setup_prefill_mode_handlers();
  } else if (mode == tt::config::LLMMode::DECODE_ONLY) {
    setup_decode_mode_handlers();
  }

  setup_common_handlers();

  TT_LOG_INFO("[SocketController] Initialized for mode: {}",
              tt::config::to_string(mode));
}

void SocketController::setup_prefill_mode_handlers() {
  socket_service_->onPrefillRequested(
      [this](const tt::sockets::PrefillRequestMessage& message) {
        TT_LOG_INFO("[SocketController] Received prefill request {}",
                    message.task_id.id);

        domain::CompletionRequest request(message.task_id);
        if (!message.token_ids.empty()) {
          std::vector<int> tokens(message.token_ids.begin(),
                                  message.token_ids.end());
          request.prompt = std::move(tokens);
        } else {
          request.prompt = message.prompt;
        }
        const std::optional<int> ORIGINAL_MAX_TOKENS = message.max_tokens;
        request.max_tokens = 1;

        auto tokenIdsPtr =
            std::make_shared<std::vector<int64_t>>(message.token_ids);
        std::string taskId = message.task_id.id;

        llm_service_->submit_streaming_request(
            request,
            [this, taskId, tokenIdsPtr, ORIGINAL_MAX_TOKENS](
                const domain::StreamingChunkResponse& chunk, bool isFinal) {
              std::string text;
              if (!chunk.choices.empty()) {
                text = chunk.choices[0].text;
                if (chunk.choices[0].token_id.has_value()) {
                  tokenIdsPtr->push_back(chunk.choices[0].token_id.value());
                }
              }

              std::optional<int> remainingTokens =
                  ORIGINAL_MAX_TOKENS.has_value()
                      ? std::optional<int>(
                            std::max(0, ORIGINAL_MAX_TOKENS.value() - 1))
                      : std::nullopt;

              tt::sockets::PrefillResultMessage msg{domain::TaskID(taskId)};
              msg.generated_text = text;
              msg.token_ids = *tokenIdsPtr;
              msg.remaining_tokens = remainingTokens;
              msg.finished = isFinal && (remainingTokens.has_value() &&
                                         remainingTokens.value() <= 0);
              msg.tokens_generated = 1;
              socket_service_->sendPrefillResult(msg);

              if (isFinal) {
                TT_LOG_INFO(
                    "[SocketController] Completed prefill {} (remaining: {}, "
                    "token_ids: {})",
                    taskId,
                    remainingTokens.has_value()
                        ? std::to_string(remainingTokens.value())
                        : "none",
                    tokenIdsPtr->size());
              }
            });
      });
}

void SocketController::setup_decode_mode_handlers() {
  llm_service_->setPrefillRequestCallback(
      [this](const domain::PrefillRequest& request) -> bool {
        return socket_service_->sendPrefillRequest(
            request.task_id, request.prompt, request.token_ids,
            request.max_tokens);
      });

  socket_service_->onPrefillComplete(
      [this](const tt::sockets::PrefillResultMessage& msg) {
        TT_LOG_INFO("[SocketController] Received prefill result {}",
                    msg.task_id.id);

        auto taken = llm_service_->detachStreamCallback(msg.task_id.id);
        if (!taken.has_value()) {
          TT_LOG_WARN("[SocketController] No callback for task_id: {}",
                      msg.task_id.id);
          return;
        }
        auto callback = std::move(taken.value());

        domain::StreamingChunkResponse response{msg.task_id};
        response.id = msg.task_id.id;
        response.created =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        domain::CompletionChoice choice;
        choice.text = msg.generated_text;
        choice.index = 0;
        response.choices.push_back(std::move(choice));
        callback(response, false);

        bool continueDecode =
            !msg.token_ids.empty() && (!msg.remaining_tokens.has_value() ||
                                       msg.remaining_tokens.value() > 0);
        if (continueDecode) {
          domain::CompletionRequest request(msg.task_id);
          std::vector<int> tokens(msg.token_ids.begin(), msg.token_ids.end());
          request.prompt = std::move(tokens);
          request.max_tokens = msg.remaining_tokens;

          llm_service_->submitDecodeContinuation(std::move(request),
                                                 std::move(callback));
        } else {
          domain::StreamingChunkResponse finalResponse{msg.task_id};
          finalResponse.id = msg.task_id.id;
          finalResponse.created =
              std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
          domain::CompletionChoice finalChoice;
          finalChoice.text = "";
          finalChoice.index = 0;
          finalChoice.finish_reason = "stop";
          finalResponse.choices.push_back(std::move(finalChoice));
          callback(finalResponse, true);
        }
      });

  socket_service_->setConnectionLostCallback([this]() {
    TT_LOG_WARN("[SocketController] Connection to prefill server lost");
    llm_service_->handleConnectionLost();
  });
}

void SocketController::setup_common_handlers() {
  socket_service_->setHealthCheckCallback([](const std::string& serverId,
                                             double /*cpu*/, double /*memory*/,
                                             int tasks) {
    TT_LOG_INFO("[SocketController] Health check from {} (active_tasks={})",
                serverId, tasks);
  });
}

}  // namespace tt::api
