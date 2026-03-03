#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_engine {

enum class DeviceBackend {
  Mock,
  Sockets,
};

enum class ModelRunnerType {
  Stub,
  Llama,
};

struct Config {
  static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 1;  // Overridden to 16 in llm_engine_config() for llama_runner
  std::vector<int64_t> stop_token_ids;  // Set by llm_engine_config() from active tokenizer strategy
  int eos = 1;
  int kvcache_block_size = 32;
  int num_kvcache_blocks = 512;
  bool reserve_first_kv_block = false;
  DeviceBackend device = DeviceBackend::Mock;
  ModelRunnerType model_runner = ModelRunnerType::Stub;
};

}  // namespace llm_engine
