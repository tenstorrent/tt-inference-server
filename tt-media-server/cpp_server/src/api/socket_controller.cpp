// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/socket_controller.hpp"
#include "domain/completion_request.hpp"
#include "services/llm_service.hpp"
#include "sockets/inter_server_service.hpp"
#include "config/settings.hpp"
#include "utils/logger.hpp"

#include <memory>

namespace tt::api {

SocketController::SocketController(
    std::shared_ptr<services::LLMService> llm_service,
    std::shared_ptr<sockets::InterServerService> socket_service)
    : llm_service_(std::move(llm_service))
    , socket_service_(std::move(socket_service)) {

    if (!socket_service_ || !socket_service_->isEnabled()) {
        TT_LOG_INFO("[SocketController] Socket service not enabled, skipping handler setup");
        return;
    }

    auto mode = tt::config::llm_mode();

    if (mode == tt::config::LLMMode::PREFILL_ONLY) {
        setup_prefill_mode_handlers();
    } else if (mode == tt::config::LLMMode::DECODE_ONLY) {
        setup_decode_mode_handlers();
    }

    setup_common_handlers();

    TT_LOG_INFO("[SocketController] Initialized for mode: {}", tt::config::to_string(mode));
}

void SocketController::setup_prefill_mode_handlers() {
    socket_service_->onPrefillRequested(
        [this](const tt::sockets::PrefillRequestMessage& message) {
            TT_LOG_INFO("[SocketController] Received prefill request {}", message.task_id.id);

            domain::CompletionRequest request(message.task_id);
            if (!message.token_ids.empty()) {
                std::vector<int> tokens(message.token_ids.begin(), message.token_ids.end());
                request.prompt = std::move(tokens);
            } else {
                request.prompt = message.prompt;
            }
            const int original_max_tokens = message.max_tokens;
            request.max_tokens = 1;

            auto token_ids_ptr = std::make_shared<std::vector<int64_t>>(message.token_ids);
            std::string task_id = message.task_id.id;

            llm_service_->submit_streaming_request(request,
                [this, task_id, token_ids_ptr, original_max_tokens]
                (const domain::StreamingChunkResponse& chunk, bool is_final) {
                    std::string text;
                    if (!chunk.choices.empty()) {
                        text = chunk.choices[0].text;
                        if (chunk.choices[0].token_id.has_value()) {
                            token_ids_ptr->push_back(chunk.choices[0].token_id.value());
                        }
                    }

                    int remaining_tokens = original_max_tokens - 1;

                    tt::sockets::PrefillResultMessage msg{domain::TaskID(task_id)};
                    msg.generated_text = text;
                    msg.token_ids = *token_ids_ptr;
                    msg.remaining_tokens = remaining_tokens;
                    msg.finished = is_final && remaining_tokens <= 0;
                    msg.tokens_generated = 1;
                    socket_service_->sendPrefillResult(msg);

                    if (is_final) {
                        TT_LOG_INFO("[SocketController] Completed prefill {} (remaining: {}, token_ids: {})",
                                    task_id, remaining_tokens, token_ids_ptr->size());
                    }
                });
        });
}

void SocketController::setup_decode_mode_handlers() {
    llm_service_->set_prefill_request_callback(
        [this](const domain::PrefillRequest& request) -> bool {
            return socket_service_->sendPrefillRequest(
                request.task_id,
                request.prompt,
                request.token_ids,
                request.max_tokens.value_or(0));  // 0 means run until EOS
        });

    socket_service_->onPrefillComplete(
        [this](const tt::sockets::PrefillResultMessage& msg) {
            TT_LOG_INFO("[SocketController] Received prefill result {}", msg.task_id.id);

            auto taken = llm_service_->detach_stream_callback(msg.task_id.id);
            if (!taken.has_value()) {
                TT_LOG_WARN("[SocketController] No callback for task_id: {}", msg.task_id.id);
                return;
            }
            auto callback = std::move(taken.value());

            domain::StreamingChunkResponse response{msg.task_id};
            response.id = msg.task_id.id;
            response.created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            domain::CompletionChoice choice;
            choice.text = msg.generated_text;
            choice.index = 0;
            response.choices.push_back(std::move(choice));
            callback(response, false);

            if (msg.remaining_tokens > 0 && !msg.token_ids.empty()) {
                domain::CompletionRequest request(msg.task_id);
                std::vector<int> tokens(msg.token_ids.begin(), msg.token_ids.end());
                request.prompt = std::move(tokens);
                request.max_tokens = msg.remaining_tokens;

                llm_service_->submit_decode_continuation(
                    std::move(request), std::move(callback));
            } else {
                domain::StreamingChunkResponse final_response{msg.task_id};
                final_response.id = msg.task_id.id;
                final_response.created = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                domain::CompletionChoice final_choice;
                final_choice.text = "";
                final_choice.index = 0;
                final_choice.finish_reason = "stop";
                final_response.choices.push_back(std::move(final_choice));
                callback(final_response, true);
            }
        });

    socket_service_->setConnectionLostCallback([this]() {
        TT_LOG_WARN("[SocketController] Connection to prefill server lost");
        llm_service_->handle_connection_lost();
    });
}

void SocketController::setup_common_handlers() {
    socket_service_->setHealthCheckCallback(
        [](const std::string& server_id, double /*cpu*/, double /*memory*/, int tasks) {
            TT_LOG_INFO("[SocketController] Health check from {} (active_tasks={})", server_id, tasks);
        });
}

} // namespace tt::api
