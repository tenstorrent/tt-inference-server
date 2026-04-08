// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "config/types.hpp"
#include "domain/llm_request.hpp"
#include "domain/llm_response.hpp"
#include "utils/concurrent_map.hpp"

namespace tt::sockets {
class InterServerService;
}

namespace tt::services {

class LLMService;

class DisaggregationService {
  using StreamCallback =
      std::function<void(const domain::LLMStreamChunk&, bool)>;

 public:
  DisaggregationService(
      tt::config::LLMMode mode, std::shared_ptr<LLMService> llmService,
      std::shared_ptr<sockets::InterServerService> socketService);
  ~DisaggregationService();

  void start();
  void stop();

  void handleStreamingRequest(domain::LLMRequest& request,
                              const StreamCallback& callback);

 private:
  void setupSocketHandlers();

  tt::config::LLMMode mode;
  std::shared_ptr<LLMService> llmService;
  std::shared_ptr<sockets::InterServerService> socketService;
  utils::ConcurrentMap<uint32_t, StreamCallback> streamCallbacks;
};

}  // namespace tt::services
