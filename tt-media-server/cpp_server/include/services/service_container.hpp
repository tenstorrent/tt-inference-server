// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "config/types.hpp"

namespace tt::services {
class DisaggregationService;
class SessionManager;
class IService;
}  // namespace tt::services

namespace tt::sockets {
class InterServerService;
}

namespace tt::services {

class ServiceContainer {
 public:
  ServiceContainer(const ServiceContainer&) = delete;
  ServiceContainer& operator=(const ServiceContainer&) = delete;

  static ServiceContainer& instance() {
    static ServiceContainer container;
    return container;
  }

  /** Wire auxiliary services that don't belong to a single ModelService.
   *  The active model service itself is installed via `registerService()`. */
  void initialize(std::shared_ptr<sockets::InterServerService> socket,
                  std::shared_ptr<DisaggregationService> disaggregation,
                  std::shared_ptr<SessionManager> sessionMgr);

  std::shared_ptr<IService> configuredService() const;

  std::shared_ptr<sockets::InterServerService> socket() const {
    return socket_;
  }
  std::shared_ptr<DisaggregationService> disaggregation() const {
    return disaggregation_;
  }
  std::shared_ptr<SessionManager> sessionManager() const {
    return sessionManager_;
  }

  /** Register a model service. Visible immediately to getService(). */
  void registerService(config::ModelService key,
                       std::shared_ptr<IService> service);
  std::shared_ptr<IService> getService(config::ModelService key) const;

 private:
  ServiceContainer() = default;

  // services_ is the single source of truth for ModelService -> IService.
  // Auxiliary slots below are not model services and stay typed.
  std::unordered_map<config::ModelService, std::shared_ptr<IService>> services_;
  std::shared_ptr<sockets::InterServerService> socket_;
  std::shared_ptr<DisaggregationService> disaggregation_;
  std::shared_ptr<SessionManager> sessionManager_;
};

}  // namespace tt::services
