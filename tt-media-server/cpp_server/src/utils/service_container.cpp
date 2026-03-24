// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/service_container.hpp"

#include <stdexcept>

#include "config/settings.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"

namespace tt::utils {

ServiceContainer* ServiceContainer::instance = nullptr;

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

void ServiceContainer::setGlobal(ServiceContainer container) {
  static ServiceContainer storage = std::move(container);
  instance = &storage;
}

const ServiceContainer& ServiceContainer::global() {
  if (!instance) {
    throw std::runtime_error(
        "[ServiceContainer] global() called before setGlobal(). "
        "Ensure initializeServices() is called in main() before Drogon "
        "starts.");
  }
  return *instance;
}

}  // namespace tt::utils
