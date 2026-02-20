#pragma once

#include <optional>

namespace llm_engine {

struct SamplingParams {
  float temperature = 1.0f;
  int max_tokens = 64;
  bool ignore_eos = false;
  std::optional<int> seed;
};

}  // namespace llm_engine
