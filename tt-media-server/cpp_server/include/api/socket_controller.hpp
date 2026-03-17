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
  SocketController(std::shared_ptr<services::LLMService> llmService,
                   std::shared_ptr<sockets::InterServerService> socketService);

  ~SocketController() = default;

  SocketController(const SocketController&) = delete;
  SocketController& operator=(const SocketController&) = delete;

 private:
  void setupPrefillModeHandlers();
  void setupDecodeModeHandlers();
  void setupCommonHandlers();

  std::shared_ptr<services::LLMService> llm_service;
  std::shared_ptr<sockets::InterServerService> socket_service;
};

}  // namespace tt::api
