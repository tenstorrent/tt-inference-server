// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/service_factory.hpp"

#include <memory>

#include "config/settings.hpp"
#include "ipc/queue_manager.hpp"
#include "profiling/tracy.hpp"
#include "services/disaggregation_service.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::service_factory {

void initializeServices() {
  tracy_config::tracyStartMainProcess();

  auto& c = services::ServiceContainer::instance();

  std::shared_ptr<services::LLMService> llm;
  std::shared_ptr<services::EmbeddingService> embedding;
  std::shared_ptr<sockets::InterServerService> socket;
  std::shared_ptr<services::DisaggregationService> disaggregation;
  std::shared_ptr<services::SessionManager> sessionManager;
  std::shared_ptr<tt::ipc::QueueManager> queueManager;

  sessionManager = std::make_shared<services::SessionManager>();

  switch (tt::config::modelService()) {
    case tt::config::ModelService::LLM: {
      queueManager = std::make_shared<tt::ipc::QueueManager>(
          static_cast<int>(tt::config::numWorkers()));
      llm = services::LLMService::createDefault(
          &tt::utils::tokenizers::activeTokenizer(), queueManager->taskQueue);
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
               std::move(disaggregation), std::move(sessionManager),
               std::move(queueManager));

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
