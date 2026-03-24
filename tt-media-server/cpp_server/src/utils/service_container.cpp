// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/service_container.hpp"

#include "config/settings.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"

namespace tt::utils {

void ServiceContainer::initialize(
    std::shared_ptr<services::LLMService> llm,
    std::shared_ptr<services::EmbeddingService> embedding,
    std::shared_ptr<sockets::InterServerService> socket,
    std::shared_ptr<services::DisaggregationService> disaggregation) {
  this->llm = std::move(llm);
  this->embedding = std::move(embedding);
  this->socket = std::move(socket);
  this->disaggregation = std::move(disaggregation);
}

std::shared_ptr<services::IService> ServiceContainer::configuredService()
    const {
  switch (tt::config::modelService()) {
    case tt::config::ModelService::LLM:
      return llm;
    case tt::config::ModelService::EMBEDDING:
      return embedding;
  }
  return nullptr;
}

}  // namespace tt::utils
