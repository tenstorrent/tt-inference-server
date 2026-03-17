// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <any>
#include <memory>
#include <typeindex>
#include <unordered_map>

#include "services/base_service.hpp"

namespace tt::utils::service_factory {

/**
 * Create and start all services determined by the current configuration.
 * Must be called early in main(), before drogon::app().run(), so that
 * worker fork()s happen in a clean process (no Drogon sockets/threads).
 */
void register_services();

namespace detail {
inline std::unordered_map<std::type_index, std::any>& service_map() {
  static std::unordered_map<std::type_index, std::any> services;
  return services;
}
}  // namespace detail

template <typename T>
void register_service(std::shared_ptr<T> service) {
  detail::service_map()[std::type_index(typeid(T))] = std::move(service);
}

/**
 * Returns the registered service of type T, or nullptr if not registered.
 * Use when you need the concrete type (e.g. LLMService, EmbeddingService).
 */
template <typename T>
std::shared_ptr<T> get_service_by_type() {
  auto& map = detail::service_map();
  auto it = map.find(std::type_index(typeid(T)));
  if (it == map.end()) return nullptr;
  return std::any_cast<std::shared_ptr<T>>(it->second);
}

/**
 * Returns the currently configured service (LLM or Embedding) as IService.
 * Use when only interface methods are needed (e.g. get_system_status).
 */
std::shared_ptr<services::IService> get_configured_service();

}  // namespace tt::utils::service_factory
