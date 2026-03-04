// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>

namespace tt::services {
class LLMService;
}

namespace tt::sockets {
class InterServerService;
}

namespace tt::api {

/**
 * @brief Controller for socket-based communication in prefill/decode split mode
 *
 * Similar to HTTP controllers, this handles the transport layer (sockets) and
 * delegates business logic to the LLMService. This keeps the service layer
 * transport-agnostic.
 *
 * Only created when LLM_MODE is "prefill" or "decode" (not "regular").
 */
class SocketController {
public:
    /**
     * @brief Construct and initialize socket controller
     * @param llm_service Reference to the LLM service for business logic
     * @param socket_service Reference to the inter-server socket service
     *
     * @note Only call this when mode is PREFILL_ONLY or DECODE_ONLY
     */
    SocketController(
        std::shared_ptr<services::LLMService> llm_service,
        std::shared_ptr<sockets::InterServerService> socket_service);

    ~SocketController() = default;

    SocketController(const SocketController&) = delete;
    SocketController& operator=(const SocketController&) = delete;

private:
    void setup_prefill_mode_handlers();
    void setup_decode_mode_handlers();
    void setup_common_handlers();

    std::shared_ptr<services::LLMService> llm_service_;
    std::shared_ptr<sockets::InterServerService> socket_service_;
};

} // namespace tt::api
