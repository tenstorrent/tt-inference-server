// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "config/types.hpp"

namespace tt::services {

class IService;

/**
 * Maps each `ModelService` to a factory that builds its `IService`, so adding
 * a new service doesn't require editing a central switch in service_factory.
 *
 * Thread-safety: registration must complete before `create()`.
 */
class ServiceRegistry {
 public:
  using ServiceFactory = std::function<std::shared_ptr<IService>()>;

  ServiceRegistry(const ServiceRegistry&) = delete;
  ServiceRegistry& operator=(const ServiceRegistry&) = delete;

  static ServiceRegistry& instance();

  /** Last write wins, so tests can override built-in factories. */
  void registerService(config::ModelService key, ServiceFactory factory);

  /** Returns nullptr if no factory is registered for `key`. */
  std::shared_ptr<IService> create(config::ModelService key) const;

  bool has(config::ModelService key) const;

  /** Test-only. */
  void clear();

 private:
  ServiceRegistry() = default;

  std::unordered_map<config::ModelService, ServiceFactory> factories_;
};

}  // namespace tt::services
