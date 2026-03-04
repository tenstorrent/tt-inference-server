// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/socket_controller.hpp"
#include "services/llm_service.hpp"
#include "sockets/inter_server_service.hpp"
#include "config/settings.hpp"

#include <iostream>

namespace tt::api {

SocketController::SocketController(
    std::shared_ptr<services::LLMService> llm_service,
    std::shared_ptr<sockets::InterServerService> socket_service)
    : llm_service_(std::move(llm_service))
    , socket_service_(std::move(socket_service)) {

    if (!socket_service_ || !socket_service_->isEnabled()) {
        std::cout << "[SocketController] Socket service not enabled, skipping handler setup\n" << std::flush;
        return;
    }

    auto mode = tt::config::llm_mode();

    if (mode == tt::config::LLMMode::PREFILL_ONLY) {
        setup_prefill_mode_handlers();
    } else if (mode == tt::config::LLMMode::DECODE_ONLY) {
        setup_decode_mode_handlers();
    }

    setup_common_handlers();

    std::cout << "[SocketController] Initialized for mode: " << tt::config::to_string(mode) << "\n" << std::flush;
}

void SocketController::setup_prefill_mode_handlers() {
    llm_service_->set_prefill_result_callback(
        [this](const domain::PrefillResult& result) {
            tt::sockets::PrefillResultMessage msg;
            msg.task_id = result.task_id;
            msg.generated_text = result.generated_text;
            msg.token_ids = result.token_ids;
            msg.remaining_tokens = result.remaining_tokens;
            msg.finished = result.finished;
            msg.tokens_generated = 1;
            socket_service_->sendPrefillResult(msg);
        });

    socket_service_->onPrefillRequested(
        [this](const tt::sockets::PrefillRequestMessage& message) {
            std::cout << "[SocketController] Received prefill request " << message.task_id << "\n" << std::flush;
            domain::PrefillRequest request;
            request.task_id = message.task_id;
            request.prompt = message.prompt;
            request.token_ids = message.token_ids;
            request.max_tokens = message.max_tokens;
            llm_service_->handle_prefill_request(request);
        });
}

void SocketController::setup_decode_mode_handlers() {
    llm_service_->set_prefill_request_callback(
        [this](const domain::PrefillRequest& request) -> bool {
            return socket_service_->sendPrefillRequest(
                request.task_id,
                request.prompt,
                request.token_ids,
                request.max_tokens);
        });

    socket_service_->onPrefillComplete(
        [this](const tt::sockets::PrefillResultMessage& msg) {
            std::cout << "[SocketController] Received prefill result " << msg.task_id << "\n" << std::flush;

            domain::PrefillResult result;
            result.task_id = msg.task_id;
            result.generated_text = msg.generated_text;
            result.token_ids = msg.token_ids;
            result.remaining_tokens = msg.remaining_tokens;
            result.finished = msg.finished;
            llm_service_->handle_prefill_complete(result);
        });
}

void SocketController::setup_common_handlers() {
    socket_service_->setConnectionLostCallback([this]() {
        std::cout << "[SocketController] Connection lost\n" << std::flush;
        llm_service_->handle_connection_lost();
    });

    socket_service_->setHealthCheckCallback(
        [](const std::string& server_id, double /*cpu*/, double /*memory*/, int tasks) {
            std::cout << "[SocketController] Health check from " << server_id
                      << " (active_tasks=" << tasks << ")\n" << std::flush;
        });
}

} // namespace tt::api
