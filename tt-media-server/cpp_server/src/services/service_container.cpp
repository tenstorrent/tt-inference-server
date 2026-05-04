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
  socket_ = std::move(socket);
  disaggregation_ = std::move(disaggregation);
  sessionManager_ = std::move(sessionMgr);

  if (llm) {
    services_[config::ModelService::LLM] =
        std::static_pointer_cast<IService>(std::move(llm));
  }
  if (embedding) {
    services_[config::ModelService::EMBEDDING] =
        std::static_pointer_cast<IService>(std::move(embedding));
  }
}

std::shared_ptr<IService> ServiceContainer::configuredService() const {
  return getService(tt::config::modelService());
}

void ServiceContainer::registerService(config::ModelService key,
                                       std::shared_ptr<IService> service) {
  services_[key] = std::move(service);
}

std::shared_ptr<IService> ServiceContainer::getService(
    config::ModelService key) const {
  auto it = services_.find(key);
  if (it == services_.end()) return nullptr;
  return it->second;
}

}  // namespace tt::services
