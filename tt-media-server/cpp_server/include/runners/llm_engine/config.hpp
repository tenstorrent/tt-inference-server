#pragma once

namespace llm_engine {

struct Config {
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 512;
  int eos = -1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = -1;

  /** When set, use real H2D/D2H sockets (tt-metal). Owned by model runner. */
  void* mesh_device = nullptr;
  void* h2d_socket = nullptr;
  void* d2h_socket = nullptr;
  /** Called after each page write to H2D to enqueue one page of device work. */
  void (*enqueue_one_page)(void* ctx) = nullptr;
  void* enqueue_one_page_ctx = nullptr;
};

}  // namespace llm_engine
