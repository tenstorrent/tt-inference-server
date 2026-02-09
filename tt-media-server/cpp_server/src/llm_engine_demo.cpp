// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/config.hpp"
#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/sampling_params.hpp"
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
#define LLM_ENGINE_DEMO_LOG_INFO LogStream("INFO")
}  // namespace

int main() {
  llm_engine::Config config;
  config.num_kvcache_blocks = 128;
  config.kvcache_block_size = 8;
  config.eos = 0;

  {
    llm_engine::LLMEngine engine{config};
    engine.add_request({1, 2, 3}, llm_engine::SamplingParams{.max_tokens = 3});
    while (!engine.is_finished()) {
      LLM_ENGINE_DEMO_LOG_INFO << "step";
      auto result = engine.step();
      for (const auto& [seq_id, tokens] : result.outputs) {
        LLM_ENGINE_DEMO_LOG_INFO << "seq " << seq_id << " completed with " << tokens.size() << " tokens";
      }
    }
    LLM_ENGINE_DEMO_LOG_INFO << "Demo done.";
  }
  return 0;
}
