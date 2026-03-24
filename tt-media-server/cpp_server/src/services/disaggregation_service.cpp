#include "services/disaggregation_service.hpp"


namespace tt::services {
  
DisaggregationService::DisaggregationService(tt::config::LLMMode mode) : mode(mode) {
    socketService = std::make_unique<tt::sockets::InterServerService>();
    socketService->initializeFromConfig();
    
    socketService->onPrefillComplete([this](const tt::sockets::PrefillResultMessage& message) {
        auto callback = streamCallbacks.get(message.task_id.id);
        if(!callback.has_value()) {
            TT_LOG_WARN("[DisaggregationService] No callback for task_id: {}", message.task_id.id);
            return;
        }
        auto response = domain::StreamingChunkResponse(message.task_id);
        response.choices.push_back(domain::CompletionChoice(message.generated_text));
        callback.value()(response, false);
        if(message.remaining_tokens.has_value() || message.remaining_tokens.value() > 0) {
            auto request = domain::CompletionRequest(message.task_id);
            request.prompt = std::vector<int>(message.token_ids.begin(), message.token_ids.end());
            request.max_tokens = message.remaining_tokens;
            llmService->submitDecodeContinuation(request, callback.value());
        } else {
          callback.value()(domain::StreamingChunkResponse(message.task_id), true);
        }
    });
    
}

DisaggregationService::~DisaggregationService() {
  socketService->stop();
}

void DisaggregationService::start() {
  if (socketService->isEnabled()) {
    socketService->start();
  }
}

void DisaggregationService::stop() {
  socketService->stop();
}

void DisaggregationService::handleStreamingRequest(domain::CompletionRequest& request, const StreamCallback& callback) {
  llmService->preProcess(request);
  streamCallbacks.insert(request.task_id.id, callback);
  auto maxTokens = request.max_tokens;
  auto tokenIds = std::get<std::vector<int>>(request.prompt);
  auto sent = socketService->sendPrefillRequest(request.task_id,
     "",
      std::vector<int64_t>(tokenIds.begin(),
       tokenIds.end()), maxTokens
  );
  if (!sent) {
    TT_LOG_ERROR("[DisaggregationService] Failed to send prefill request for task_id: {}", request.task_id.id);
    streamCallbacks.erase(request.task_id.id);
    throw std::runtime_error("[DisaggregationService] Failed to send prefill request for task_id: " + request.task_id.id);
  }
}

}  // namespace tt::services