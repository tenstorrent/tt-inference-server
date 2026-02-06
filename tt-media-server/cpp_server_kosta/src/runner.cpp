#include "runner.hpp"
#include <chrono>
#include <thread>
#include <vector>

// Mock tokens for simulated generation
static const std::vector<std::string> MOCK_TOKENS = {
    "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."
};

ModelRunner::ModelRunner(const std::string &model_path) {
  // No initialization needed for mock runner
}

domain::Response ModelRunner::run(const domain::Request &req) {
  domain::Response res;
  res.id = req.id;
  res.is_finished = true;

  // Generate mock response with delays
  std::string result;
  int num_tokens = req.max_tokens;
  for (int i = 0; i < num_tokens; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(24));
    result += MOCK_TOKENS[i % MOCK_TOKENS.size()];
  }
  res.setResult(result);

  return res;
}

void ModelRunner::run_streaming(
    const domain::Request &req,
    std::function<void(const domain::Response &)> callback) {

  int num_tokens = req.max_tokens ;
  if (num_tokens <= 0) num_tokens = 1;
  
  for (int i = 0; i < num_tokens; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(24));
    
    domain::Response res;
    res.id = req.id;
    res.is_finished = (i == num_tokens - 1);
    res.setResult(MOCK_TOKENS[i % MOCK_TOKENS.size()]);
    callback(res);
  }
}

// Phase 2: Batched streaming - reduces IPC calls by batching tokens
// Sends first token immediately for low TTFT, then batches remaining tokens
void ModelRunner::run_streaming_batched(
    const domain::Request &req,
    std::function<void(const domain::BatchedResponse &)> callback) {

  int num_tokens = req.max_tokens;
  if (num_tokens <= 0) num_tokens = 1;
  
  // Send first token immediately for low TTFT
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(24));
    domain::BatchedResponse first;
    first.request_id = req.id;
    first.addToken(MOCK_TOKENS[0]);
    first.is_finished = (num_tokens == 1);
    callback(first);
  }
  
  if (num_tokens == 1) return;
  
  // Batch remaining tokens starting from index 1
  domain::BatchedResponse batch;
  batch.request_id = req.id;
  
  for (int i = 1; i < num_tokens; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(24));
    
    batch.addToken(MOCK_TOKENS[i % MOCK_TOKENS.size()]);
    batch.is_finished = (i == num_tokens - 1);
    
    // Send when batch is full OR this is the last token
    if (batch.isFull() || batch.is_finished) {
      callback(batch);
      batch.reset();  // Reset for next batch
    }
  }
}
