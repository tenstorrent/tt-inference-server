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
  int max_num_batched_tokens = 64 * MAX_INPUT_TOKENS;
  int max_num_seqs = 1;
  std::vector<int64_t> stop_token_ids;  // Set by llm_engine_config() from active tokenizer strategy
  int eos = 1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = 1024;
  ModelRunnerType runner_type = ModelRunnerType::Mock;
};

}  // namespace llm_engine
