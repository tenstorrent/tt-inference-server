#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_engine {

enum class ModelRunnerType {
  Mock,
  TtRun,
  Llama
};

struct Config {
  static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 1;  // Overridden to 16 in llm_engine_config() for llama_runner
  std::vector<int64_t> stop_token_ids;  // Set by llm_engine_config() from active tokenizer strategy
  int eos = 1;
  int kvcache_block_size = 32;
  int num_kvcache_blocks = 512;
  ModelRunnerType runner_type = ModelRunnerType::Mock;
};

}  // namespace llm_engine
