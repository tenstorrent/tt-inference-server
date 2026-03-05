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
  int num_kvcache_blocks = 512;
  ModelRunnerType runner_type = ModelRunnerType::Mock;
};

/**
 * Model-specific configuration profiles.
 * Each profile defines the optimal settings for a particular model.
 */

/**
 * DeepSeek R1 0528 configuration.
 */
struct DeepseekConfig {
  static constexpr int max_num_batched_tokens = 64 * Config::MAX_INPUT_TOKENS;
  static constexpr int max_num_seqs = 1;
  static constexpr int kvcache_block_size = 256;
  static constexpr int num_kvcache_blocks = 512;
  static constexpr ModelRunnerType runner_type = ModelRunnerType::Mock;

  static Config create() {
    Config cfg;
    cfg.max_num_batched_tokens = max_num_batched_tokens;
    cfg.max_num_seqs = max_num_seqs;
    cfg.kvcache_block_size = kvcache_block_size;
    cfg.num_kvcache_blocks = num_kvcache_blocks;
    cfg.runner_type = runner_type;
    return cfg;
  }
};

/**
 * Llama 3.1 8B Instruct configuration.
 */
struct LlamaConfig {
  static constexpr int max_num_batched_tokens = 16384;
  static constexpr int max_num_seqs = 16;
  static constexpr int kvcache_block_size = 32;
  static constexpr int num_kvcache_blocks = 512;
  static constexpr ModelRunnerType runner_type = ModelRunnerType::Llama;

  static Config create() {
    Config cfg;
    cfg.max_num_batched_tokens = max_num_batched_tokens;
    cfg.max_num_seqs = max_num_seqs;
    cfg.kvcache_block_size = kvcache_block_size;
    cfg.num_kvcache_blocks = num_kvcache_blocks;
    cfg.runner_type = runner_type;
    return cfg;
  }
};

}  // namespace llm_engine
