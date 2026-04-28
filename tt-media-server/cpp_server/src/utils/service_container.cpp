// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/service_container.hpp"

#include "config/settings.hpp"
#include "services/conversation_store.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"

namespace tt::utils {

void ServiceContainer::initialize(
    std::shared_ptr<services::LLMService> llm,
    std::shared_ptr<services::EmbeddingService> embedding,
    std::shared_ptr<sockets::InterServerService> socket,
    std::shared_ptr<services::DisaggregationService> disaggregation,
    std::shared_ptr<services::SessionManager> sessionMgr,
    std::shared_ptr<services::ConversationStore> conversationStore) {
  llm_ = std::move(llm);
  embedding_ = std::move(embedding);
  socket_ = std::move(socket);
  disaggregation_ = std::move(disaggregation);
  sessionManager_ = std::move(sessionMgr);
  conversationStore_ = std::move(conversationStore);
}

std::shared_ptr<services::IService> ServiceContainer::configuredService()
    const {
  switch (tt::config::modelService()) {
    case tt::config::ModelService::LLM:
      return llm_;
    case tt::config::ModelService::EMBEDDING:
      return embedding_;
  }
  return nullptr;
}

}  // namespace tt::utils
