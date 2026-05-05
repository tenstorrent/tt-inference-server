// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/service_container.hpp"

#include "config/settings.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"

namespace tt::services {

void ServiceContainer::initialize(
    std::shared_ptr<LLMService> llm,
    std::shared_ptr<EmbeddingService> embedding,
    std::shared_ptr<sockets::InterServerService> socket,
    std::shared_ptr<DisaggregationService> disaggregation,
    std::shared_ptr<SessionManager> sessionMgr) {
  llm_ = std::move(llm);
  embedding_ = std::move(embedding);
  socket_ = std::move(socket);
  disaggregation_ = std::move(disaggregation);
  sessionManager_ = std::move(sessionMgr);
}

std::shared_ptr<IService> ServiceContainer::configuredService() const {
  switch (tt::config::modelService()) {
    case tt::config::ModelService::LLM:
      return llm_;
    case tt::config::ModelService::EMBEDDING:
      return embedding_;
  }
  return nullptr;
}

}  // namespace tt::services
