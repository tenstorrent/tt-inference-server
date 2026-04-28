// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <stdexcept>

namespace tt::services {
class LLMService;
class EmbeddingService;
class DisaggregationService;
class SessionManager;
class ConversationStore;
class IService;
}  // namespace tt::services

namespace tt::sockets {
class InterServerService;
}

namespace tt::utils {

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
  void initialize(
      std::shared_ptr<services::LLMService> llm,
      std::shared_ptr<services::EmbeddingService> embedding,
      std::shared_ptr<sockets::InterServerService> socket,
      std::shared_ptr<services::DisaggregationService> disaggregation,
      std::shared_ptr<services::SessionManager> sessionMgr,
      std::shared_ptr<services::ConversationStore> conversationStore);

  std::shared_ptr<services::IService> configuredService() const;

  std::shared_ptr<services::LLMService> llm() const { return llm_; }
  std::shared_ptr<services::EmbeddingService> embedding() const {
    return embedding_;
  }
  std::shared_ptr<sockets::InterServerService> socket() const {
    return socket_;
  }
  std::shared_ptr<services::DisaggregationService> disaggregation() const {
    return disaggregation_;
  }
  std::shared_ptr<services::SessionManager> sessionManager() const {
    return sessionManager_;
  }
  std::shared_ptr<services::ConversationStore> conversationStore() const {
    return conversationStore_;
  }

 private:
  ServiceContainer() = default;

  std::shared_ptr<services::LLMService> llm_;
  std::shared_ptr<services::EmbeddingService> embedding_;
  std::shared_ptr<sockets::InterServerService> socket_;
  std::shared_ptr<services::DisaggregationService> disaggregation_;
  std::shared_ptr<services::SessionManager> sessionManager_;
  std::shared_ptr<services::ConversationStore> conversationStore_;
};

}  // namespace tt::utils
