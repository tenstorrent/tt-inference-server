#pragma once

#include <cstdint>
#include <vector>

namespace llm_engine {

enum class DeviceBackend {
  Mock,
  Sockets,
};

struct Config {
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 32;
  std::vector<int64_t> stop_token_ids;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = 512;
  DeviceBackend device = DeviceBackend::Mock;
};

}  // namespace llm_engine
