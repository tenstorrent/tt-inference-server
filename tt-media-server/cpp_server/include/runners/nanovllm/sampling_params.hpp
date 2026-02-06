#pragma once

namespace nanovllm {

struct SamplingParams {
  float temperature = 1.0f;
  int max_tokens = 64;
  bool ignore_eos = false;
};

}  // namespace nanovllm
