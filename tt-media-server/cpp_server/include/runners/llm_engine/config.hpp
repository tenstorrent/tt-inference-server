#pragma once

#include <cstdint>
#include <vector>

namespace llm_engine {

struct Config {
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 32;
  std::vector<int64_t> stop_token_ids;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = 512;
};

}  // namespace llm_engine
