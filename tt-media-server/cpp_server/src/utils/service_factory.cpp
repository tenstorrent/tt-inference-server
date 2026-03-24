// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/service_factory.hpp"

#include <memory>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/disaggregation_service.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"

namespace tt::utils::service_factory {

ServiceContainer buildServices() {
  ServiceContainer c;

  if (tt::config::isLlmServiceEnabled()) {
    c.llm = std::make_shared<services::LLMService>();

    auto mode = tt::config::llmMode();
    if (mode != tt::config::LLMMode::REGULAR) {
      c.socket = std::make_shared<sockets::InterServerService>();
      c.socket->initializeFromConfig();
      c.disaggregation = std::make_shared<services::DisaggregationService>(
          mode, c.llm, c.socket);
    }
  }

  if (tt::config::isEmbeddingService()) {
    c.embedding = std::make_shared<services::EmbeddingService>();
  }

  return c;
}

void startServices(ServiceContainer& c) {
  tracy_config::tracyStartMainProcess();

  if (c.llm) {
    c.llm->start();
    TT_LOG_INFO("[ServiceFactory] LLM service started");
  }
  if (c.disaggregation) {
    c.disaggregation->start();
    TT_LOG_INFO("[ServiceFactory] Disaggregation service started");
  }
  if (c.embedding) {
    c.embedding->start();
    TT_LOG_INFO("[ServiceFactory] Embedding service started");
  }
}

void initializeServices() {
  auto container = buildServices();
  startServices(container);
  ServiceContainer::setGlobal(std::move(container));
}

}  // namespace tt::utils::service_factory
