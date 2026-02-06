#pragma once
#include <cstring>
#include <string>

namespace domain {

constexpr size_t MAX_PROMPT_SIZE = 4096;
constexpr size_t MAX_RESULT_SIZE = 8192;

// Phase 2: Batched token constants
// Batch size of 4 balances latency (96ms max) vs IPC overhead (4x reduction)
constexpr size_t MAX_BATCH_TOKENS = 4;
constexpr size_t MAX_TOKEN_SIZE = 128;  // Reduced from 256 - most tokens are small

struct Request {
  char prompt[MAX_PROMPT_SIZE];
  int id;
  int max_tokens = 100;

  // Helper to set prompt from std::string
  void setPrompt(const std::string &s) {
    std::strncpy(prompt, s.c_str(), MAX_PROMPT_SIZE - 1);
    prompt[MAX_PROMPT_SIZE - 1] = '\0';
  }
};

struct Response {
  char result[MAX_RESULT_SIZE];
  int id;
  bool is_finished = false;

  // Helper to set result from std::string
  void setResult(const std::string &s) {
    std::strncpy(result, s.c_str(), MAX_RESULT_SIZE - 1);
    result[MAX_RESULT_SIZE - 1] = '\0';
  }
};

// Phase 2: Batched response for reduced IPC overhead
// Instead of sending one token per message, batch up to MAX_BATCH_TOKENS
struct BatchedResponse {
  int request_id;
  int num_tokens;
  bool is_finished;
  char tokens[MAX_BATCH_TOKENS][MAX_TOKEN_SIZE];
  
  BatchedResponse() : request_id(0), num_tokens(0), is_finished(false) {
    std::memset(tokens, 0, sizeof(tokens));
  }
  
  void addToken(const std::string& token) {
    if (num_tokens < static_cast<int>(MAX_BATCH_TOKENS)) {
      std::strncpy(tokens[num_tokens], token.c_str(), MAX_TOKEN_SIZE - 1);
      tokens[num_tokens][MAX_TOKEN_SIZE - 1] = '\0';
      num_tokens++;
    }
  }
  
  void reset() {
    num_tokens = 0;
    is_finished = false;
  }
  
  bool isFull() const {
    return num_tokens >= static_cast<int>(MAX_BATCH_TOKENS);
  }
};

} // namespace domain
