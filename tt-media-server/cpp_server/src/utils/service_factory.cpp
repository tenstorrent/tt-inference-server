// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/service_factory.hpp"

#include <memory>

#include "api/route_registry.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/disaggregation_service.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"
#include "services/modality_registration.hpp"
#include "services/service_registry.hpp"
#include "services/session_manager.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"

namespace tt::utils::service_factory {

namespace {

void logActiveModality() {
  const auto active = tt::config::modelService();
  TT_LOG_INFO("[ServiceFactory] Active modality: {}",
              tt::config::toString(active));
  for (const auto& route : api::RouteRegistry::instance().routesFor(active)) {
    TT_LOG_INFO("  {} {}  - {}", route.method, route.path, route.description);
  }
  for (const auto& path : api::RouteRegistry::instance().alwaysExemptPaths()) {
    TT_LOG_INFO("  *      {}  - always available", path);
  }
}

}  // namespace

void initializeServices() {
  tracy_config::tracyStartMainProcess();

  // Populate the (Service, Runner, Route) registries. Idempotent.
  services::registerBuiltinModalities();

  auto& c = services::ServiceContainer::instance();
  const auto active = tt::config::modelService();

  std::shared_ptr<services::LLMService> llm;
  std::shared_ptr<services::EmbeddingService> embedding;
  std::shared_ptr<sockets::InterServerService> socket;
  std::shared_ptr<services::DisaggregationService> disaggregation;
  auto sessionManager = std::make_shared<services::SessionManager>();

  // Build the service for the active modality through the registry. We then
  // downcast into the typed slots that LLMController / EmbeddingController
  // continue to consume; future modalities will populate only the generic
  // slot via ServiceContainer::registerService().
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

  // Register the activeService into the generic slot for any modality that
  // doesn't have a typed accessor on the container.
  if (activeService && !c.getService(active)) {
    c.registerService(active, activeService);
  }

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

  logActiveModality();
}

}  // namespace tt::utils::service_factory
