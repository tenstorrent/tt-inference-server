#include <string>
#include <cstdint>

namespace api {

// OpenAI v1/completions compatible request
struct Request {
  std::string model;
  std::string prompt;
  int max_tokens = 16;
  double temperature = 1.0;
  double top_p = 1.0;
  int n = 1;
  bool stream = false;
  std::string stop;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::string user;
};

// OpenAI v1/completions compatible response (used for formatting)
struct Response {
  std::string id;
  std::string object = "text_completion";
  int64_t created;
  std::string model;
  std::string text;
  std::string finish_reason = "stop";
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
};

} // namespace api
