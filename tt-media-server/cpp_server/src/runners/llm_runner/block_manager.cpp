#include "runners/llm_runner/block_manager.hpp"

#include <algorithm>
#include <stdexcept>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/hash.hpp"

namespace llm_engine {

Block::Block(int blockId) : block_id(blockId) {}

void Block::update(int64_t hash, std::vector<int64_t> tokenIds) {
  this->hash = hash;
  this->token_ids = std::move(tokenIds);
}

void Block::reset() {
  ref_count = 1;
  hash = -1;
  token_ids.clear();
}

BlockManager::BlockManager(size_t numBlocks, size_t blockSize)
    : block_size_(blockSize) {
  if (numBlocks == 0) {
    throw std::invalid_argument("BlockManager: num_blocks must be positive");
  }
  blocks_.reserve(numBlocks);
  for (size_t i = 0; i < numBlocks; ++i) {
    blocks_.emplace_back(static_cast<int>(i));
  }
  for (size_t i = 0; i < numBlocks; ++i) {
    free_block_ids_.push_back(static_cast<int>(i));
  }
}

int64_t BlockManager::computeHash(const std::vector<int64_t>& tokenIds,
                                  int64_t prefix) {
  return hashTokenIds(tokenIds, prefix);
}

Block& BlockManager::allocateBlock(int blockId) {
  ZoneScopedN("BlockManager::allocate_block");
  Block& block = blocks_[static_cast<size_t>(blockId)];
  if (block.ref_count != 0) {
    throw std::logic_error(
        "BlockManager::allocateBlock: block " + std::to_string(blockId) +
        " has non-zero ref_count " + std::to_string(block.ref_count));
  }
  block.reset();
  free_block_ids_.erase(
      std::find(free_block_ids_.begin(), free_block_ids_.end(), blockId));
  used_block_ids_.insert(blockId);
  LLM_ENGINE_LOG("block_manager")
      << "allocate_block block_id=" << blockId
      << " free=" << free_block_ids_.size()
      << " used=" << used_block_ids_.size() << std::endl;
  return blocks_[static_cast<size_t>(blockId)];
}

void BlockManager::deallocateBlock(int blockId) {
  ZoneScopedN("BlockManager::deallocate_block");
  if (blocks_[static_cast<size_t>(blockId)].ref_count != 0) {
    throw std::logic_error("BlockManager::deallocateBlock: block " +
                           std::to_string(blockId) +
                           " still has non-zero ref_count");
  }
  used_block_ids_.erase(blockId);
  free_block_ids_.push_back(blockId);
  LLM_ENGINE_LOG("block_manager")
      << "deallocate_block block_id=" << blockId
      << " free=" << free_block_ids_.size() << std::endl;
}

size_t BlockManager::numFreeBlocks() const {
  std::lock_guard<std::mutex> lock(mutex);
  return free_block_ids_.size();
}

bool BlockManager::allocate(Sequence& seq) {
  ZoneScopedN("BlockManager::allocate");
  std::lock_guard<std::mutex> lock(mutex);
  if (!seq.blockTable.empty()) {
    throw std::logic_error(
        "BlockManager::allocate: sequence already has blocks allocated");
  }

  if (free_block_ids_.size() < seq.numBlocks()) {
    return false;
  }

  LLM_ENGINE_LOG("block_manager")
      << "allocate task_id=" << seq.taskId << " num_blocks=" << seq.numBlocks()
      << " free=" << free_block_ids_.size() << std::endl;
  int64_t h = -1;
  bool cacheMiss = false;
  for (size_t i = 0; i < seq.numBlocks(); ++i) {
    std::vector<int64_t> tokenIds = seq.block(i);
    h = (tokenIds.size() == block_size_) ? computeHash(tokenIds, h) : -1;
    auto it = hash_to_block_id_.find(h);
    int blockId = (it != hash_to_block_id_.end()) ? it->second : -1;
    if (blockId == -1 ||
        blocks_[static_cast<size_t>(blockId)].token_ids != tokenIds) {
      cacheMiss = true;
    }
    if (cacheMiss) {
      blockId = free_block_ids_.front();
      Block& block = allocateBlock(blockId);
      if (h != -1) {
        block.update(h, tokenIds);
        hash_to_block_id_[h] = blockId;
      }
      seq.blockTable.push_back(blockId);
    } else {
      seq.numCachedTokens += block_size_;
      if (used_block_ids_.count(blockId)) {
        blocks_[static_cast<size_t>(blockId)].ref_count += 1;
      } else {
        Block& block = allocateBlock(blockId);
        if (h != -1) {
          block.update(h, tokenIds);
          hash_to_block_id_[h] = blockId;
        }
      }
      seq.blockTable.push_back(blockId);
    }
  }
  return true;
}

void BlockManager::deallocate(Sequence& seq) {
  ZoneScopedN("BlockManager::deallocate");
  std::lock_guard<std::mutex> lock(mutex);
  LLM_ENGINE_LOG("block_manager")
      << "deallocate task_id=" << seq.taskId
      << " num_blocks=" << seq.blockTable.size() << std::endl;
  for (auto it = seq.blockTable.rbegin(); it != seq.blockTable.rend(); ++it) {
    int blockId = *it;
    Block& block = blocks_[static_cast<size_t>(blockId)];
    block.ref_count -= 1;
    if (block.ref_count == 0) {
      deallocateBlock(blockId);
    }
  }
  seq.numCachedTokens = 0;
  seq.blockTable.clear();
}

bool BlockManager::canAppend(const Sequence& seq) const {
  std::lock_guard<std::mutex> lock(mutex);
  size_t needOne = (seq.size() % block_size_ == 1) ? 1 : 0;
  return free_block_ids_.size() >= needOne;
}

void BlockManager::mayAppend(Sequence& seq) {
  ZoneScopedN("BlockManager::may_append");
  std::lock_guard<std::mutex> lock(mutex);
  std::vector<int>& blockTable = seq.blockTable;
  Block& lastBlock = blocks_[static_cast<size_t>(blockTable.back())];
  size_t len = seq.size();
  if (len % block_size_ == 1) {
    LLM_ENGINE_LOG("block_manager") << "may_append task_id=" << seq.taskId
                                    << " new_block len=" << len << std::endl;
    if (lastBlock.hash == -1) {
      throw std::logic_error(
          "BlockManager::mayAppend: expected last block to be hashed");
    }
    int blockId = free_block_ids_.front();
    allocateBlock(blockId);
    blockTable.push_back(blockId);
  } else if (len % block_size_ == 0) {
    if (lastBlock.hash != -1) {
      throw std::logic_error(
          "BlockManager::mayAppend: expected last block to be unhashed");
    }
    LLM_ENGINE_LOG("block_manager")
        << "may_append task_id=" << seq.taskId << " fill_last_block len=" << len
        << std::endl;
    std::vector<int64_t> tokenIds = seq.block(seq.numBlocks() - 1);
    int64_t prefix =
        (blockTable.size() > 1)
            ? blocks_[static_cast<size_t>(blockTable[blockTable.size() - 2])]
                  .hash
            : -1;
    int64_t h = computeHash(tokenIds, prefix);
    lastBlock.update(h, tokenIds);
    hash_to_block_id_[h] = lastBlock.block_id;
  } else {
    if (lastBlock.hash != -1) {
      throw std::logic_error(
          "BlockManager::mayAppend: expected last block to be unhashed");
    }
  }
}

}  // namespace llm_engine
