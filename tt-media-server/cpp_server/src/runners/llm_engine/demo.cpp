#include "llm_engine/config.hpp"
#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/sampling_params.hpp"
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <vector>

int main() {
  llm_engine::Config config;
  config.num_kvcache_blocks = 128;
  config.kvcache_block_size = 8;
  config.eos = 0;

  {
    llm_engine::LLMEngine engine{config};
    engine.add_request({1, 2, 3}, llm_engine::SamplingParams{.max_tokens = 2});
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    llm_engine::StepResultCallback on_step;
    on_step = [&](llm_engine::StepResult result) {
      std::cout << "step\n";
      for (const auto& [seq_id, tokens] : result.outputs) {
        std::cout << "seq " << seq_id << " completed with " << tokens.size()
                  << " tokens\n";
      }
      if (engine.is_finished()) {
        std::lock_guard lock{m};
        done = true;
        cv.notify_one();
      } else {
        engine.step(on_step);
      }
    };
    engine.step(on_step);
    std::unique_lock lock{m};
    cv.wait(lock, [&] { return done; });
    std::cout << "Demo done.\n";
  }
  return 0;
}
