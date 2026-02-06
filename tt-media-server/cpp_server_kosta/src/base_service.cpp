#include "base_service.hpp"

BaseService &BaseService::getInstace() {
  static BaseService instance;
  return instance;
}

void BaseService::process(const api::Request &req, drogon::ResponseStreamPtr stream) {
  int key = tracker.put(std::move(stream)); // request tracker generates a request key(id) for us
  domain::Request request;
  request.setPrompt(req.prompt);
  request.id = key;
  request.max_tokens = req.max_tokens;
  scheduler.put(request);
}
