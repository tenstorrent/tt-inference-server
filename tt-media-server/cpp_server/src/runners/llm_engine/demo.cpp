#include "llm_engine/config.hpp"
#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/sampling_params.hpp"
#include <iostream>
#include "llm_engine/engine/in_memory_task_queue.hpp"

int main() {
  llm_engine::Config config;
  config.num_kvcache_blocks = 128;
  config.kvcache_block_size = 8;
  config.eos = 0;

  int finished_count = 0;
  constexpr int TOTAL_REQUESTS = 3;
  
  std::unique_ptr<llm_engine::Scheduler> scheduler = std::make_unique<llm_engine::Scheduler>(config, std::make_unique<llm_engine::InMemoryTaskQueue>());

  llm_engine::LLMEngine engine{config, [&](int seq_id, int64_t token_id, bool finished) {
    std::cout << "seq " << seq_id << " token " << token_id;
    if (finished) {
      std::cout << " [done]";
      if (++finished_count == TOTAL_REQUESTS) engine.stop();
    }
    std::cout << "\n";
  }, std::move(scheduler)};

  engine.scheduler().add_request({1, 2, 3}, {.max_tokens = 30});
  engine.scheduler().add_request({4, 5, 6, 7}, {.max_tokens = 10});
  engine.scheduler().add_request({7, 8, 9, 10, 11, 12}, {.max_tokens = 20});

  engine.run();

  std::cout << "Demo done.\n";
  return 0;
}
