#pragma once

#include <memory>
#include "sockets/inter_server_service.hpp"
#include "domain/prefill_request.hpp"
#include "config/types.hpp"
#include "services/llm_service.hpp"
#include "utils/concurrent_map.hpp"

namespace tt::services {
    
using StreamCallback = std::function<void(domain::StreamingChunkResponse&, bool)>;

class DisaggregationService: public IService {
 public:
  DisaggregationService(tt::config::LLMMode mode);
  ~DisaggregationService();

  void start();
  void stop();
  
  bool sendPrefillRequest(const domain::PrefillRequest& request) const;
  
  void handleStreamingRequest(domain::CompletionRequest& request, const StreamCallback& callback);
  
 private:
  std::unique_ptr<tt::sockets::InterServerService> socketService;
  std::unique_ptr<LLMService> llmService;
  tt::config::LLMMode mode;
  ConcurrentMap<std::string, StreamCallback> streamCallbacks;
};

}  // namespace tt::services