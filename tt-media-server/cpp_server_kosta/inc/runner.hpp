#pragma once
#include "domain.hpp"
#include <functional>
#include <string>

class ModelRunner {
public:
  ModelRunner(const std::string &model_path);
  
  // Non-streaming: returns complete response
  domain::Response run(const domain::Request &req);
  
  // Streaming: calls callback for each token (legacy)
  void run_streaming(
      const domain::Request &req,
      std::function<void(const domain::Response &)> callback);
  
  // Phase 2: Batched streaming - sends tokens in batches for reduced IPC overhead
  void run_streaming_batched(
      const domain::Request &req,
      std::function<void(const domain::BatchedResponse &)> callback);
};
