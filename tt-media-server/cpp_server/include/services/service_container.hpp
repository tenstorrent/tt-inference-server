// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "config/types.hpp"

namespace tt::services {
class LLMService;
class EmbeddingService;
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

  /** Wire services created for the active MODEL_SERVICE; pass nullptr for any
   *  slot not used in the current mode. */
  void initialize(std::shared_ptr<LLMService> llm,
                  std::shared_ptr<EmbeddingService> embedding,
                  std::shared_ptr<sockets::InterServerService> socket,
                  std::shared_ptr<DisaggregationService> disaggregation,
                  std::shared_ptr<SessionManager> sessionMgr);

  std::shared_ptr<IService> configuredService() const;

  std::shared_ptr<LLMService> llm() const { return llm_; }
  std::shared_ptr<EmbeddingService> embedding() const { return embedding_; }
  std::shared_ptr<sockets::InterServerService> socket() const {
    return socket_;
  }
  std::shared_ptr<DisaggregationService> disaggregation() const {
    return disaggregation_;
  }
  std::shared_ptr<SessionManager> sessionManager() const {
    return sessionManager_;
  }

  /** Generic slot for modalities without a typed accessor. LLM/Embedding
   *  are mirrored into this map for backwards-compatible lookup. */
  void registerService(config::ModelService key,
                       std::shared_ptr<IService> service);
  std::shared_ptr<IService> getService(config::ModelService key) const;

 private:
  ServiceContainer() = default;

  std::shared_ptr<LLMService> llm_;
  std::shared_ptr<EmbeddingService> embedding_;
  std::shared_ptr<sockets::InterServerService> socket_;
  std::shared_ptr<DisaggregationService> disaggregation_;
  std::shared_ptr<SessionManager> sessionManager_;
  std::unordered_map<config::ModelService, std::shared_ptr<IService>> services_;
};

}  // namespace tt::services
