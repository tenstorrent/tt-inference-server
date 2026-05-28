// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/block_hash_accumulator.hpp"

#include "config/settings.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace tt::utils {

BlockHashAccumulator::BlockHashAccumulator(
    std::vector<uint64_t> initialHashes,
    std::vector<int> partialBlockTokens)
    : hashes_(std::move(initialHashes)),
      blockBuffer_(std::move(partialBlockTokens)),
      firstBlockSize_(config::kvCacheFirstBlockSize()),
      blockSize_(config::kvCacheBlockSize()) {
  parentHash_ = hashes_.empty() ? 0 : hashes_.back();
}

std::optional<std::vector<uint64_t>> BlockHashAccumulator::addToken(
    int tokenId) {
  blockBuffer_.push_back(tokenId);

  size_t targetSize = hashes_.empty() ? firstBlockSize_ : blockSize_;
  if (blockBuffer_.size() < targetSize) {
    return std::nullopt;
  }

  // Block complete - compute hash using chained seeding
  // Must use sizeof(int) to match conversation_hasher.cpp hashing
  uint64_t newHash = XXH64(blockBuffer_.data(),
                           blockBuffer_.size() * sizeof(int), parentHash_);
  hashes_.push_back(newHash);
  parentHash_ = newHash;
  blockBuffer_.clear();

  return hashes_;
}

}  // namespace tt::utils
