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
 * Registry mapping each ModelService to a factory that builds its IService.
 * Used by `service_factory::initializeServices()` so adding a new model
 * service doesn't require editing a central switch.
 *
 * Thread-safety: registration must complete before `create()` is called.
 */
class ServiceRegistry {
 public:
  using ServiceFactory = std::function<std::shared_ptr<IService>()>;

  ServiceRegistry(const ServiceRegistry&) = delete;
  ServiceRegistry& operator=(const ServiceRegistry&) = delete;

  static ServiceRegistry& instance();

  /** Register a factory for a given model service. Last write wins, so
   *  callers may override built-in factories in tests. */
  void registerService(config::ModelService key, ServiceFactory factory);

  /** Construct the service for `key`. Returns nullptr if no factory is
   *  registered. */
  std::shared_ptr<IService> create(config::ModelService key) const;

  /** True iff a factory is registered for `key`. */
  bool has(config::ModelService key) const;

  /** Remove all registrations. Test-only helper. */
  void clear();

 private:
  ServiceRegistry() = default;

  std::unordered_map<config::ModelService, ServiceFactory> factories_;
};

}  // namespace tt::services
