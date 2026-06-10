// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/service_registry.hpp"

#include "services/request_pipeline.hpp"

namespace tt::services {

ServiceRegistry& ServiceRegistry::instance() {
  static ServiceRegistry registry;
  return registry;
}

void ServiceRegistry::registerService(config::ModelService key,
                                      ServiceFactory factory) {
  factories_[key] = std::move(factory);
}

std::shared_ptr<IService> ServiceRegistry::create(
    config::ModelService key) const {
  auto it = factories_.find(key);
  if (it == factories_.end() || !it->second) return nullptr;
  return it->second();
}

bool ServiceRegistry::has(config::ModelService key) const {
  auto it = factories_.find(key);
  return it != factories_.end() && static_cast<bool>(it->second);
}

void ServiceRegistry::clear() { factories_.clear(); }

}  // namespace tt::services
