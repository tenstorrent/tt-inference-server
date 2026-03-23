#include "services/disaggregation_service.hpp"

namespace tt::services {

DisaggregationService::DisaggregationService(tt::config::LLMMode mode) : mode(mode) {
    socketService = std::make_unique<tt::sockets::InterServerService>();
    socketService->initializeFromConfig();
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

bool DisaggregationService::sendPrefillRequest(const domain::PrefillRequest& request) const {

}

}  // namespace tt::services