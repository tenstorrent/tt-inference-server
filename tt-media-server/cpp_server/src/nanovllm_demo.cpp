// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "nanovllm/config.hpp"
#include "nanovllm/engine/llm_engine.hpp"
#include "nanovllm/sampling_params.hpp"
#include <sstream>
#include <iostream>

namespace {
struct LogStream {
    std::ostringstream ss;
    const char* level;
    LogStream(const char* l) : level(l) {}
    ~LogStream() { std::cout << "[" << level << "] " << ss.str() << std::endl; }
    template<typename T>
    LogStream& operator<<(const T& v) { ss << v; return *this; }
};
#define NANOVLLM_DEMO_LOG_INFO LogStream("INFO")
}  // namespace

int main() {
  nanovllm::Config config;
  config.num_kvcache_blocks = 128;
  config.kvcache_block_size = 8;
  config.eos = 0;

  nanovllm::LLMEngine engine{config};
  engine.add_request({1, 2, 3}, nanovllm::SamplingParams{.max_tokens = 30});
  engine.add_request({4, 5, 6, 7}, nanovllm::SamplingParams{.max_tokens = 10});
  engine.add_request({7, 8, 9, 10, 11, 12}, nanovllm::SamplingParams{.max_tokens = 20});
  while (!engine.is_finished()) {
    NANOVLLM_DEMO_LOG_INFO << "step";
    auto result = engine.step();
    for (const auto& [seq_id, tokens] : result.outputs) {
      NANOVLLM_DEMO_LOG_INFO << "seq " << seq_id << " completed with " << tokens.size() << " tokens";
    }
  }
  NANOVLLM_DEMO_LOG_INFO << "Demo done.";
  return 0;
}
