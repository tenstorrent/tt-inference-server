#pragma once

namespace nanovllm {

struct Config {
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 512;
  int eos = -1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = -1;
};

}  // namespace nanovllm
