#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
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

  bool allocate(Sequence& seq);
  void deallocate(Sequence& seq);
  bool canAppend(const Sequence& seq) const;
  void mayAppend(Sequence& seq);

  int blockSize() const { return block_size_; }
  int numFreeBlocks() const;

 private:
  Block& allocateBlock(int blockId);
  void deallocateBlock(int blockId);

  int block_size_;
  std::vector<Block> blocks_;
  std::unordered_map<int64_t, int> hash_to_block_id_;
  std::deque<int> free_block_ids_;
  std::unordered_set<int> used_block_ids_;
  mutable std::mutex mutex;
};

}  // namespace llm_engine
