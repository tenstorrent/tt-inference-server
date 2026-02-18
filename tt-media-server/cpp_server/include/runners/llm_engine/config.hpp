#pragma once

namespace llm_engine {

struct Config {
  int max_num_batched_tokens = 16384;
  static constexpr int max_num_seqs = 1; // Temporary hardcoded value for Deepseek blitz decode
  int eos = -1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = -1;
  bool use_real_device = false;
};

}  // namespace llm_engine
