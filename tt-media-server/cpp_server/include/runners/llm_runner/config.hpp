#pragma once

#include <cstddef>

namespace llm_engine {

enum class DeviceBackend {
  Mock,
  Sockets,
  TtRun,
};

struct Config {
  static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
  int max_num_batched_tokens = 64 * MAX_INPUT_TOKENS;
  static constexpr int max_num_seqs = 1; // Temporary hardcoded value for Deepseek blitz decode
  int eos = 1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = 512;
  DeviceBackend device = DeviceBackend::Mock;
};

}  // namespace llm_engine
