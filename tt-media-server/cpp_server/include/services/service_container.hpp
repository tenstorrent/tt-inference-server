// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
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

namespace tt::api::resolvers {
class ChatCompletionsResolver;
class ResponsesResolver;
}  // namespace tt::api::resolvers

namespace tt::services {

class ServiceContainer {
 public:
  ServiceContainer(const ServiceContainer&) = delete;
  ServiceContainer& operator=(const ServiceContainer&) = delete;

  static ServiceContainer& instance() {
    static ServiceContainer container;
    return container;
  }

  /** Wire auxiliary services. The active model service itself is installed
   *  via `registerService()`. */
  void initialize(std::shared_ptr<sockets::InterServerService> socket,
                  std::shared_ptr<DisaggregationService> disaggregation,
                  std::shared_ptr<SessionManager> sessionMgr,
                  std::shared_ptr<api::resolvers::ChatCompletionsResolver>
                      chatCompletionsResolver,
                  std::shared_ptr<api::resolvers::ResponsesResolver>
                      responsesResolver);

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
  std::shared_ptr<api::resolvers::ChatCompletionsResolver>
  chatCompletionsResolver() const {
    return chatCompletionsResolver_;
  }
  std::shared_ptr<api::resolvers::ResponsesResolver> responsesResolver() const {
    return responsesResolver_;
  }

  void registerService(config::ModelService key,
                       std::shared_ptr<IService> service);
  std::shared_ptr<IService> getService(config::ModelService key) const;

 private:
  ServiceContainer() = default;

  // Single source of truth for ModelService -> IService; auxiliary slots
  // below are not model services and stay typed.
  std::unordered_map<config::ModelService, std::shared_ptr<IService>> services_;
  std::shared_ptr<sockets::InterServerService> socket_;
  std::shared_ptr<DisaggregationService> disaggregation_;
  std::shared_ptr<SessionManager> sessionManager_;
  std::shared_ptr<api::resolvers::ChatCompletionsResolver>
      chatCompletionsResolver_;
  std::shared_ptr<api::resolvers::ResponsesResolver> responsesResolver_;
};

}  // namespace tt::services
