#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_engine {

enum class ModelRunnerType {
  Mock,
  Pipeline,
  Llama
};

enum class SchedulingPolicy {
  PREFILL_FIRST,
  MAX_OCCUPANCY,
};

struct Config {
  static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
  int max_num_batched_tokens = 64 * MAX_INPUT_TOKENS;
  int max_num_seqs = 1;
  int max_in_flight_count = 64;
  std::vector<int64_t> stop_token_ids;  // Set by llm_engine_config() from active tokenizer strategy
  int eos = 1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = 512;
  ModelRunnerType runner_type = ModelRunnerType::Mock;
  SchedulingPolicy scheduling_policy = SchedulingPolicy::PREFILL_FIRST;
};

}  // namespace llm_engine
