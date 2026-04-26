#include "runners/llm_runner/block_manager.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/hash.hpp"
#include "utils/logger.hpp"

namespace tt::runners::llm_engine {
using Sequence = tt::domain::Sequence;

Block::Block(int blockId) : blockId(blockId) {}

void Block::update(int64_t hash, std::vector<int64_t> tokenIds) {
  this->hash = hash;
  this->tokenIds = std::move(tokenIds);
}

void Block::reset() {
  refCount = 1;
  hash = -1;
  tokenIds.clear();
}

BlockManager::BlockManager(size_t numBlocks, size_t blockSize)
    : blockSize(blockSize) {
  if (numBlocks == 0) {
    throw std::invalid_argument("BlockManager: num_blocks must be positive");
  }
  blocks.reserve(numBlocks);
  for (size_t i = 0; i < numBlocks; ++i) {
    blocks.emplace_back(static_cast<int>(i));
  }
  for (size_t i = 0; i < numBlocks; ++i) {
    freeBlockIds.push_back(static_cast<int>(i));
  }
}

int64_t BlockManager::computeHash(const std::vector<int64_t>& tokenIds,
                                  int64_t prefix) {
  return hashTokenIds(tokenIds, prefix);
}

Block& BlockManager::allocateBlock(int blockId) {
  ZoneScopedN("BlockManager::allocate_block");
  Block& block = blocks[static_cast<size_t>(blockId)];
  assert(block.refCount == 0);
  block.reset();
  freeBlockIds.erase(
      std::find(freeBlockIds.begin(), freeBlockIds.end(), blockId));
  usedBlockIds.insert(blockId);
  TT_LOG_DEBUG("[block_manager] allocate_block block_id={} free={} used={}",
               blockId, freeBlockIds.size(), usedBlockIds.size());
  return blocks[static_cast<size_t>(blockId)];
}

void BlockManager::deallocateBlock(int blockId) {
  ZoneScopedN("BlockManager::deallocate_block");
  assert(blocks[static_cast<size_t>(blockId)].refCount == 0);
  usedBlockIds.erase(blockId);
  freeBlockIds.push_back(blockId);
  TT_LOG_DEBUG("[block_manager] deallocate_block block_id={} free={}", blockId,
               freeBlockIds.size());
}

size_t BlockManager::numFreeBlocks() const {
  std::lock_guard<std::mutex> lock(mutex);
  return freeBlockIds.size();
}

bool BlockManager::allocate(Sequence& seq) {
  ZoneScopedN("BlockManager::allocate");
  std::lock_guard<std::mutex> lock(mutex);
  assert(seq.getBlockTable().empty());

  if (freeBlockIds.size() < seq.numBlocks()) {
    return false;
  }

  TT_LOG_DEBUG("[block_manager] allocate task_id={} num_blocks={} free={}",
               seq.taskId, seq.numBlocks(), freeBlockIds.size());
  int64_t h = -1;
  bool cacheMiss = false;
  for (size_t i = 0; i < seq.numBlocks(); ++i) {
    std::vector<int64_t> tokenIds = seq.block(i);
    h = (tokenIds.size() == blockSize) ? computeHash(tokenIds, h) : -1;
    auto it = hashToBlockId.find(h);
    int blockId = (it != hashToBlockId.end()) ? it->second : -1;
    if (blockId == -1 ||
        blocks[static_cast<size_t>(blockId)].tokenIds != tokenIds) {
      cacheMiss = true;
    }
    if (cacheMiss) {
      blockId = freeBlockIds.front();
      Block& block = allocateBlock(blockId);
      if (h != -1) {
        block.update(h, tokenIds);
        hashToBlockId[h] = blockId;
      }
      seq.getMutableBlockTable().push_back(blockId);
    } else {
      seq.setNumCachedTokens(seq.getNumCachedTokens() + blockSize);
      if (usedBlockIds.count(blockId)) {
        blocks[static_cast<size_t>(blockId)].refCount += 1;
      } else {
        Block& block = allocateBlock(blockId);
        if (h != -1) {
          block.update(h, tokenIds);
          hashToBlockId[h] = blockId;
        }
      }
      seq.getMutableBlockTable().push_back(blockId);
    }
  }
  return true;
}

void BlockManager::deallocate(Sequence& seq) {
  ZoneScopedN("BlockManager::deallocate");
  std::lock_guard<std::mutex> lock(mutex);
  TT_LOG_DEBUG("[block_manager] deallocate task_id={} num_blocks={}",
               seq.taskId, seq.getBlockTable().size());
  for (auto it = seq.getBlockTable().rbegin(); it != seq.getBlockTable().rend();
       ++it) {
    int blockId = *it;
    Block& block = blocks[static_cast<size_t>(blockId)];
    block.refCount -= 1;
    if (block.refCount == 0) {
      deallocateBlock(blockId);
    }
  }
  seq.setNumCachedTokens(0);
  seq.getMutableBlockTable().clear();
}

bool BlockManager::canAppend(const Sequence& seq) const {
  std::lock_guard<std::mutex> lock(mutex);
  size_t needOne = (seq.size() % blockSize == 1) ? 1 : 0;
  return freeBlockIds.size() >= needOne;
}

void BlockManager::mayAppend(Sequence& seq) {
  ZoneScopedN("BlockManager::may_append");
  std::lock_guard<std::mutex> lock(mutex);
  std::vector<int>& blockTable = seq.getMutableBlockTable();
  Block& lastBlock = blocks[static_cast<size_t>(blockTable.back())];
  size_t len = seq.size();
  if (len % blockSize == 1) {
    TT_LOG_DEBUG("[block_manager] may_append task_id={} new_block len={}",
                 seq.taskId, len);
    assert(lastBlock.hash != -1);
    int blockId = freeBlockIds.front();
    allocateBlock(blockId);
    blockTable.push_back(blockId);
  } else if (len % blockSize == 0) {
    assert(lastBlock.hash == -1);
    TT_LOG_DEBUG("[block_manager] may_append task_id={} fill_last_block len={}",
                 seq.taskId, len);
    std::vector<int64_t> tokenIds = seq.block(seq.numBlocks() - 1);
    int64_t prefix =
        (blockTable.size() > 1)
            ? blocks[static_cast<size_t>(blockTable[blockTable.size() - 2])]
                  .hash
            : -1;
    int64_t h = computeHash(tokenIds, prefix);
    lastBlock.update(h, tokenIds);
    hashToBlockId[h] = lastBlock.blockId;
  } else {
    assert(lastBlock.hash == -1);
  }
}

}  // namespace tt::runners::llm_engine
