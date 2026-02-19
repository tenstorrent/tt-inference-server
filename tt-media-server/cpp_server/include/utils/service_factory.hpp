// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <any>
#include <memory>
#include <typeindex>
#include <unordered_map>

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
} // namespace detail

template<typename T>
void register_service(std::shared_ptr<T> service) {
    detail::service_map()[std::type_index(typeid(T))] = std::move(service);
}

template<typename T>
std::shared_ptr<T> get_service() {
    auto& map = detail::service_map();
    auto it = map.find(std::type_index(typeid(T)));
    if (it == map.end()) return nullptr;
    return std::any_cast<std::shared_ptr<T>>(it->second);
}

} // namespace tt::utils::service_factory
