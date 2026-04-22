#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "domain/sequence.hpp"

namespace tt::runners::llm_engine {

class Block {
 public:
  explicit Block(int blockId);

  void update(int64_t hash, std::vector<int64_t> tokenIds);
  void reset();

  int blockId = 0;
  int refCount = 0;
  int64_t hash = -1;
  std::vector<int64_t> tokenIds;
};

class BlockManager {
 public:
  BlockManager(size_t numBlocks, size_t blockSize);

  static int64_t computeHash(const std::vector<int64_t>& tokenIds,
                             int64_t prefix = -1);

  bool allocate(tt::domain::Sequence& seq);
  void deallocate(tt::domain::Sequence& seq);
  bool canAppend(const tt::domain::Sequence& seq) const;
  void mayAppend(tt::domain::Sequence& seq);

  int getBlockSize() const { return static_cast<int>(blockSize); }
  size_t numFreeBlocks() const;

 private:
  Block& allocateBlock(int blockId);
  void deallocateBlock(int blockId);

  size_t blockSize;
  std::vector<Block> blocks;
  std::unordered_map<int64_t, int> hashToBlockId;
  std::deque<int> freeBlockIds;
  std::unordered_set<int> usedBlockIds;
  mutable std::mutex mutex;
};

}  // namespace tt::runners::llm_engine
