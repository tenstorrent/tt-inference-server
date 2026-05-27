// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/block_hash_accumulator.hpp"

#include "config/settings.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace tt::utils {

BlockHashAccumulator::BlockHashAccumulator(
    std::vector<uint64_t> initialHashes,
    std::vector<int64_t> partialBlockTokens)
    : hashes_(std::move(initialHashes)),
      blockBuffer_(std::move(partialBlockTokens)),
      firstBlockSize_(config::kvCacheFirstBlockSize()),
      blockSize_(config::kvCacheBlockSize()) {
  parentHash_ = hashes_.empty() ? 0 : hashes_.back();
}

std::optional<std::vector<uint64_t>> BlockHashAccumulator::addToken(
    int64_t tokenId) {
  blockBuffer_.push_back(tokenId);

  size_t targetSize = hashes_.empty() ? firstBlockSize_ : blockSize_;
  if (blockBuffer_.size() < targetSize) {
    return std::nullopt;
  }

  // Block complete - compute hash using chained seeding
  uint64_t newHash = XXH64(blockBuffer_.data(),
                           blockBuffer_.size() * sizeof(int64_t), parentHash_);
  hashes_.push_back(newHash);
  parentHash_ = newHash;
  blockBuffer_.clear();

  return hashes_;
}

std::vector<uint64_t> BlockHashAccumulator::finalize() {
  if (blockBuffer_.empty()) {
    return hashes_;
  }

  // Hash the partial block (even if incomplete)
  uint64_t newHash = XXH64(blockBuffer_.data(),
                           blockBuffer_.size() * sizeof(int64_t), parentHash_);
  hashes_.push_back(newHash);
  blockBuffer_.clear();

  return hashes_;
}

}  // namespace tt::utils
