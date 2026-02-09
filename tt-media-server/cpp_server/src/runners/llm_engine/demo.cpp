#include "llm_engine/config.hpp"
#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/sampling_params.hpp"
#include <iostream>
#include <vector>

int main() {
  llm_engine::Config config;
  config.num_kvcache_blocks = 128;
  config.kvcache_block_size = 8;
  config.eos = 0;

  {
    llm_engine::LLMEngine engine{config};
    engine.add_request({1, 2, 3}, llm_engine::SamplingParams{.max_tokens = 2});
    while (!engine.is_finished()) {
      std::cout << "step\n";
      auto result = engine.step();
      for (const auto& [seq_id, tokens] : result.outputs) {
        std::cout << "seq " << seq_id << " completed with " << tokens.size()
                  << " tokens\n";
      }
    }
    std::cout << "Demo done.\n";
  }
  return 0;
}
