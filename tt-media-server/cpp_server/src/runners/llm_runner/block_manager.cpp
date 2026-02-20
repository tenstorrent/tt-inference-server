#include "runners/llm_runner/block_manager.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/hash.hpp"
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace llm_engine {

Block::Block(int block_id) : block_id(block_id) {}

void Block::update(int64_t hash, std::vector<int64_t> token_ids) {
  this->hash = hash;
  this->token_ids = std::move(token_ids);
}

void Block::reset() {
  ref_count = 1;
  hash = -1;
  token_ids.clear();
}

BlockManager::BlockManager(int num_blocks, int block_size)
    : block_size_(block_size) {
  if (num_blocks <= 0) {
    throw std::invalid_argument(
        "BlockManager: num_blocks must be positive, got " + std::to_string(num_blocks));
  }
  blocks_.reserve(static_cast<size_t>(num_blocks));
  for (int i = 0; i < num_blocks; ++i) {
    blocks_.emplace_back(i);
  }
  for (int i = 0; i < num_blocks; ++i) {
    free_block_ids_.push_back(i);
  }
}

int64_t BlockManager::compute_hash(const std::vector<int64_t>& token_ids,
                                  int64_t prefix) {
  return hash_token_ids(token_ids, prefix);
}

Block& BlockManager::allocate_block(int block_id) {
  Block& block = blocks_[static_cast<size_t>(block_id)];
  assert(block.ref_count == 0);
  block.reset();
  free_block_ids_.erase(
      std::find(free_block_ids_.begin(), free_block_ids_.end(), block_id));
  used_block_ids_.insert(block_id);
  LLM_ENGINE_LOG("block_manager") << "allocate_block block_id=" << block_id
                                 << " free=" << free_block_ids_.size()
                                 << " used=" << used_block_ids_.size() << std::endl;
  return blocks_[static_cast<size_t>(block_id)];
}

void BlockManager::deallocate_block(int block_id) {
  assert(blocks_[static_cast<size_t>(block_id)].ref_count == 0);
  used_block_ids_.erase(block_id);
  free_block_ids_.push_back(block_id);
  LLM_ENGINE_LOG("block_manager") << "deallocate_block block_id=" << block_id
                                 << " free=" << free_block_ids_.size() << std::endl;
}

bool BlockManager::can_allocate(const Sequence& seq) const {
  return static_cast<int>(free_block_ids_.size()) >=
         static_cast<int>(seq.num_blocks());
}

void BlockManager::allocate(Sequence& seq) {
  assert(seq.block_table_.empty());
  LLM_ENGINE_LOG("block_manager") << "allocate task_id=" << seq.task_id
                                << " num_blocks=" << seq.num_blocks()
                                << " free=" << free_block_ids_.size() << std::endl;
  int64_t h = -1;
  bool cache_miss = false;
  for (size_t i = 0; i < seq.num_blocks(); ++i) {
    std::vector<int64_t> token_ids = seq.block(i);
    h = (token_ids.size() == static_cast<size_t>(block_size_))
            ? compute_hash(token_ids, h)
            : -1;
    auto it = hash_to_block_id_.find(h);
    int block_id = (it != hash_to_block_id_.end()) ? it->second : -1;
    if (block_id == -1 ||
        blocks_[static_cast<size_t>(block_id)].token_ids != token_ids) {
      cache_miss = true;
    }
    if (cache_miss) {
      block_id = free_block_ids_.front();
      Block& block = allocate_block(block_id);
      if (h != -1) {
        block.update(h, token_ids);
        hash_to_block_id_[h] = block_id;
      }
      seq.block_table_.push_back(block_id);
    } else {
      seq.num_cached_tokens_ += static_cast<size_t>(block_size_);
      if (used_block_ids_.count(block_id)) {
        blocks_[static_cast<size_t>(block_id)].ref_count += 1;
      } else {
        Block& block = allocate_block(block_id);
        if (h != -1) {
          block.update(h, token_ids);
          hash_to_block_id_[h] = block_id;
        }
      }
      seq.block_table_.push_back(block_id);
    }
  }
}

void BlockManager::deallocate(Sequence& seq) {
  LLM_ENGINE_LOG("block_manager") << "deallocate task_id=" << seq.task_id
                                 << " num_blocks=" << seq.block_table_.size() << std::endl;
  for (auto it = seq.block_table_.rbegin(); it != seq.block_table_.rend();
       ++it) {
    int block_id = *it;
    Block& block = blocks_[static_cast<size_t>(block_id)];
    block.ref_count -= 1;
    if (block.ref_count == 0) {
      deallocate_block(block_id);
    }
  }
  seq.num_cached_tokens_ = 0;
  seq.block_table_.clear();
}

bool BlockManager::can_append(const Sequence& seq) const {
  int need_one = (seq.size() % block_size_ == 1) ? 1 : 0;
  return static_cast<int>(free_block_ids_.size()) >= need_one;
}

void BlockManager::may_append(Sequence& seq) {
  std::vector<int>& block_table = seq.block_table_;
  Block& last_block = blocks_[static_cast<size_t>(block_table.back())];
  size_t len = seq.size();
  if (len % static_cast<size_t>(block_size_) == 1) {
    LLM_ENGINE_LOG("block_manager") << "may_append task_id=" << seq.task_id
                                  << " new_block len=" << len << std::endl;
    assert(last_block.hash != -1);
    int block_id = free_block_ids_.front();
    allocate_block(block_id);
    block_table.push_back(block_id);
  } else if (len % static_cast<size_t>(block_size_) == 0) {
    assert(last_block.hash == -1);
    LLM_ENGINE_LOG("block_manager") << "may_append task_id=" << seq.task_id
                                  << " fill_last_block len=" << len << std::endl;
    std::vector<int64_t> token_ids = seq.block(seq.num_blocks() - 1);
    int64_t prefix = (block_table.size() > 1)
                         ? blocks_[static_cast<size_t>(block_table[block_table.size() - 2])].hash
                         : -1;
    int64_t h = compute_hash(token_ids, prefix);
    last_block.update(h, token_ids);
    hash_to_block_id_[h] = last_block.block_id;
  } else {
    assert(last_block.hash == -1);
  }
}

}  // namespace llm_engine
