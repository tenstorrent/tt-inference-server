// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cassert>
#include <memory>

namespace tt::services {
class LLMService;
class EmbeddingService;
class DisaggregationService;
class IService;
}  // namespace tt::services

namespace tt::sockets {
class InterServerService;
}

namespace tt::utils {

struct ServiceContainer {
  std::shared_ptr<services::LLMService> llm;
  std::shared_ptr<services::EmbeddingService> embedding;
  std::shared_ptr<sockets::InterServerService> socket;
  std::shared_ptr<services::DisaggregationService> disaggregation;

  std::shared_ptr<services::IService> configuredService() const;

  static void setGlobal(ServiceContainer container);
  static const ServiceContainer& global();

 private:
  static ServiceContainer* instance;
};

}  // namespace tt::utils
