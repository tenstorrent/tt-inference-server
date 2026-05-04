// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/service_factory.hpp"

#include <memory>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/disaggregation_service.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"
#include "services/model_service_registration.hpp"
#include "services/service_registry.hpp"
#include "services/session_manager.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"

namespace tt::utils::service_factory {

void initializeServices() {
  tracy_config::tracyStartMainProcess();

  services::registerBuiltinModelServices();

  auto& c = services::ServiceContainer::instance();
  const auto active = tt::config::modelService();

  std::shared_ptr<services::LLMService> llm;
  std::shared_ptr<services::EmbeddingService> embedding;
  std::shared_ptr<sockets::InterServerService> socket;
  std::shared_ptr<services::DisaggregationService> disaggregation;
  auto sessionManager = std::make_shared<services::SessionManager>();

  // Build the active service through the registry, then downcast into the
  // typed slots that LLMController / EmbeddingController still consume. New
  // model services populate only the generic slot below.
  auto activeService = services::ServiceRegistry::instance().create(active);
  if (!activeService) {
    TT_LOG_WARN(
        "[ServiceFactory] No service registered for MODEL_SERVICE='{}'; "
        "container left empty.",
        tt::config::toString(active));
  }

  switch (active) {
    case tt::config::ModelService::LLM: {
      llm = std::dynamic_pointer_cast<services::LLMService>(activeService);
      if (!llm) {
        TT_LOG_ERROR(
            "[ServiceFactory] LLM factory produced unexpected service type");
      }
      const auto mode = tt::config::llmMode();
      if (llm && mode != tt::config::LLMMode::REGULAR) {
        socket = std::make_shared<sockets::InterServerService>();
        socket->initializeFromConfig();
        disaggregation = std::make_shared<services::DisaggregationService>(
            mode, llm, socket);
      }
      break;
    }
    case tt::config::ModelService::EMBEDDING:
      embedding =
          std::dynamic_pointer_cast<services::EmbeddingService>(activeService);
      if (!embedding) {
        TT_LOG_ERROR(
            "[ServiceFactory] Embedding factory produced unexpected service "
            "type");
      }
      break;
  }

  c.initialize(std::move(llm), std::move(embedding), std::move(socket),
               std::move(disaggregation), std::move(sessionManager));

  // Mirror the active service into the generic map for model services that
  // don't flow through the typed `initialize()` parameters (no-op for LLM /
  // Embedding, which initialize() already inserts).
  if (activeService && !c.getService(active)) {
    c.registerService(active, activeService);
  }

  if (auto svc = c.getService(active)) {
    svc->start();
    TT_LOG_INFO("[ServiceFactory] {} service started",
                tt::config::toString(active));
  }
  if (c.disaggregation()) {
    c.disaggregation()->start();
    TT_LOG_INFO("[ServiceFactory] Disaggregation service started");
  }
  if (c.sessionManager()) {
    TT_LOG_INFO("[ServiceFactory] Session manager initialized");
  }

  TT_LOG_INFO("[ServiceFactory] Active model service: {}",
              tt::config::toString(active));
}

}  // namespace tt::utils::service_factory
