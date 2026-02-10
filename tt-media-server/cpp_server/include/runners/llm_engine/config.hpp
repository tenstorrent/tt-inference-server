#pragma once

#include <string>

namespace llm_engine {

struct Config {
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 512;
  int eos = -1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = -1;

  /// IPC task queue name (e.g. "/tt_task_queue"). Required.
  std::string task_queue_name;
  /// Maximum number of messages the IPC task queue can hold.
  int task_queue_capacity = 1024;
};

}  // namespace llm_engine
