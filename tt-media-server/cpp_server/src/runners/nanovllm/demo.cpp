#include "nanovllm/config.hpp"
#include "nanovllm/engine/llm_engine.hpp"
#include "nanovllm/sampling_params.hpp"
#include <iostream>
#include <vector>

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
    std::cout << "step\n";
    auto result = engine.step();
    for (const auto& [seq_id, tokens] : result.outputs) {
      std::cout << "seq " << seq_id << " completed with " << tokens.size()
                << " tokens\n";
    }
  }
  std::cout << "Demo done.\n";
  return 0;
}
