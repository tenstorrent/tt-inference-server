// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/service_factory.hpp"
#include <memory>

#include "api/socket_controller.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"
#include "utils/logger.hpp"

namespace tt::utils::service_factory {

namespace {
std::unique_ptr<tt::api::SocketController> socketController;
}

void registerServices() {
  tracy_config::tracyStartMainProcess();

  if (tt::config::isLlmServiceEnabled()) {
    auto llm = std::make_shared<services::LLMService>();
    llm->start();

    if (tt::config::llmMode() != tt::config::LLMMode::REGULAR) {
      auto socketService = std::make_shared<tt::sockets::InterServerService>();
      socketService->initializeFromConfig();
      if (socketService->isEnabled()) {
        socketService->start();
      }
      socketController = std::make_unique<tt::api::SocketController>(
          llm, socketService);
    }

    registerService(llm);
    TT_LOG_INFO("[ServiceFactory] LLM service registered and started");
  }

  if (tt::config::isEmbeddingService()) {
    auto emb = std::make_shared<services::EmbeddingService>();
    emb->start();
    registerService(std::move(emb));
    TT_LOG_INFO("[ServiceFactory] Embedding service registered and started");
  }
}

std::shared_ptr<services::IService> getConfiguredService() {
  switch (tt::config::modelService()) {
    case tt::config::ModelService::LLM:
      return getServiceByType<services::LLMService>();
    case tt::config::ModelService::EMBEDDING:
      return getServiceByType<services::EmbeddingService>();
  }
  return nullptr;
}

}  // namespace tt::utils::service_factory
