#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runners/llm_runner/sequence.hpp"

namespace llm_engine {

class Block {
 public:
  explicit Block(int blockId);

  void update(int64_t hash, std::vector<int64_t> tokenIds);
  void reset();

  int block_id = 0;
  int ref_count = 0;
  int64_t hash = -1;
  std::vector<int64_t> token_ids;
};

class BlockManager {
 public:
  BlockManager(int numBlocks, int blockSize);

  static int64_t computeHash(const std::vector<int64_t>& tokenIds,
                             int64_t prefix = -1);

  bool canAllocate(const Sequence& seq) const;
  void allocate(Sequence& seq);
  void deallocate(Sequence& seq);
  bool canAppend(const Sequence& seq) const;
  void mayAppend(Sequence& seq);

 private:
  Block& allocateBlock(int blockId);
  void deallocateBlock(int blockId);

  int block_size;
  std::vector<Block> blocks;
  std::unordered_map<int64_t, int> hash_to_block_id;
  std::deque<int> free_block_ids;
  std::unordered_set<int> used_block_ids;
};

}  // namespace llm_engine
