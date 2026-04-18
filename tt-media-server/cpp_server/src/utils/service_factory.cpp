// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/service_factory.hpp"

#include <memory>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/disaggregation_service.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"

namespace tt::utils::service_factory {

void initializeServices() {
  tracy_config::tracyStartMainProcess();

  auto& c = ServiceContainer::instance();

  std::shared_ptr<services::LLMService> llm;
  std::shared_ptr<services::EmbeddingService> embedding;
  std::shared_ptr<sockets::InterServerService> socket;
  std::shared_ptr<services::DisaggregationService> disaggregation;
  std::shared_ptr<services::SessionManager> sessionManager;

  // Create SessionManager for all modes
  sessionManager = std::make_shared<services::SessionManager>();

  // Only construct services for MODEL_SERVICE (see config::modelService()).
  // Additional modes (e.g. videogen) extend config::ModelService and add cases.
  switch (tt::config::modelService()) {
    case tt::config::ModelService::LLM: {
      llm = std::make_shared<services::LLMService>();
      auto mode = tt::config::llmMode();
      if (mode != tt::config::LLMMode::REGULAR) {
        socket = std::make_shared<sockets::InterServerService>();
        socket->initializeFromConfig();
        disaggregation = std::make_shared<services::DisaggregationService>(
            mode, llm, socket);
      }
      break;
    }
    case tt::config::ModelService::EMBEDDING:
      embedding = std::make_shared<services::EmbeddingService>();
      break;
  }

  c.initialize(std::move(llm), std::move(embedding), std::move(socket),
               std::move(disaggregation), std::move(sessionManager));

  if (c.llm()) {
    c.llm()->start();
    TT_LOG_INFO("[ServiceFactory] LLM service started");
  }
  if (c.disaggregation()) {
    c.disaggregation()->start();
    TT_LOG_INFO("[ServiceFactory] Disaggregation service started");
  }
  if (c.embedding()) {
    c.embedding()->start();
    TT_LOG_INFO("[ServiceFactory] Embedding service started");
  }
  if (c.sessionManager()) {
    TT_LOG_INFO("[ServiceFactory] Session manager initialized");
  }
}

}  // namespace tt::utils::service_factory
