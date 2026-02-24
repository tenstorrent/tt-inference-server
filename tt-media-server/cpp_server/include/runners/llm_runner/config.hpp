#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_engine {

enum class DeviceBackend {
  Mock,
  Sockets,
};

struct Config {
  static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 16;
  std::vector<int64_t> stop_token_ids;  // Set by llm_engine_config() from active tokenizer strategy
  int eos = 1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = 512;
  DeviceBackend device = DeviceBackend::Mock;
};

}  // namespace llm_engine
