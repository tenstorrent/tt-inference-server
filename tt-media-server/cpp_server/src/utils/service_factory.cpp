// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/service_factory.hpp"

#include <memory>

#include "api/resolvers/chat_completions_resolver.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_service.hpp"
#include "services/model_service_registration.hpp"
#include "services/service_registry.hpp"
#include "services/session_manager.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"

namespace tt::utils::service_factory {

namespace {

// Per-model-service post-construction wiring. New services that need
// auxiliaries add an `if (auto x = dynamic_pointer_cast<XService>(...))` arm
// next to the LLM one rather than growing a central switch.
struct AuxiliaryServices {
  std::shared_ptr<sockets::InterServerService> socket;
  std::shared_ptr<services::DisaggregationService> disaggregation;
};

AuxiliaryServices buildAuxiliaryServices(
    const std::shared_ptr<services::IService>& activeService) {
  if (auto llm =
          std::dynamic_pointer_cast<services::LLMService>(activeService)) {
    const auto mode = tt::config::llmMode();
    if (mode != tt::config::LLMMode::REGULAR) {
      auto socket = std::make_shared<sockets::InterServerService>();
      socket->initializeFromConfig();
      auto disagg =
          std::make_shared<services::DisaggregationService>(mode, llm, socket);
      return {std::move(socket), std::move(disagg)};
    }
  }
  return {};
}

}  // namespace

void initializeServices() {
  tracy_config::tracyStartMainProcess();

  services::registerBuiltinModelServices();

  auto& c = services::ServiceContainer::instance();
  const auto active = tt::config::modelService();

  auto activeService = services::ServiceRegistry::instance().create(active);
  if (!activeService) {
    TT_LOG_WARN(
        "[ServiceFactory] No service registered for MODEL_SERVICE='{}'; "
        "container left empty.",
        tt::config::toString(active));
  } else {
    c.registerService(active, activeService);
  }

  auto aux = buildAuxiliaryServices(activeService);

  auto sessionManager = std::make_shared<services::SessionManager>();
  auto chatCompletionsResolver =
      std::make_shared<api::resolvers::ChatCompletionsResolver>(sessionManager);

  c.initialize(std::move(aux.socket), std::move(aux.disaggregation),
               std::move(sessionManager), std::move(chatCompletionsResolver));

  if (auto svc = c.configuredService()) {
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
