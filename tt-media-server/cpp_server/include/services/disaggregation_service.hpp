// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "config/types.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "utils/concurrent_map.hpp"

namespace tt::sockets {

using namespace tt::domain::llm;
class InterServerService;
}  // namespace tt::sockets

namespace tt::services {

using namespace tt::domain::llm;

class LLMService;

class DisaggregationService {
  using StreamCallback = std::function<void(const LLMStreamChunk&, bool)>;

 public:
  DisaggregationService(
      tt::config::LLMMode mode, std::shared_ptr<LLMService> llmService,
      std::shared_ptr<sockets::InterServerService> socketService);
  ~DisaggregationService();

  void start();
  void stop();

  void handleStreamingRequest(LLMRequest& request,
                              const std::vector<uint64_t>& registrationHashes,
                              const StreamCallback& callback);
  void abortRequest(uint32_t taskId);

 private:
  void setupSocketHandlers();

  tt::config::LLMMode mode;
  std::shared_ptr<LLMService> llmService;
  std::shared_ptr<sockets::InterServerService> socketService;
  utils::ConcurrentMap<uint32_t, StreamCallback> streamCallbacks;
};

}  // namespace tt::services
