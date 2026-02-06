#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

class Block {
 public:
  explicit Block(int block_id);

  void update(int64_t hash, std::vector<int64_t> token_ids);
  void reset();

  int block_id = 0;
  int ref_count = 0;
  int64_t hash = -1;
  std::vector<int64_t> token_ids;
};

class BlockManager {
 public:
  BlockManager(int num_blocks, int block_size);

  static int64_t compute_hash(const std::vector<int64_t>& token_ids,
                              int64_t prefix = -1);

  bool can_allocate(const Sequence& seq) const;
  void allocate(Sequence& seq);
  void deallocate(Sequence& seq);
  bool can_append(const Sequence& seq) const;
  void may_append(Sequence& seq);

 private:
  Block& allocate_block(int block_id);
  void deallocate_block(int block_id);

  int block_size_;
  std::vector<Block> blocks_;
  std::unordered_map<int64_t, int> hash_to_block_id_;
  std::deque<int> free_block_ids_;
  std::unordered_set<int> used_block_ids_;
};

}  // namespace llm_engine
