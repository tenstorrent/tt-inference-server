// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace tt::utils {

/**
 * Accumulates tokens during streaming and computes block hashes incrementally.
 *
 * Uses vLLM-style chained xxHash64: each block's hash uses the previous
 * block's hash as the seed. Returns the updated hash vector whenever a
 * complete block is formed.
 */
class BlockHashAccumulator {
 public:
  /**
   * @param initialHashes Block hashes already computed from the prompt.
   * @param partialBlockTokens Tokens from the prompt's incomplete final block
   *        (if any). Empty if prompt ended on a block boundary.
   */
  BlockHashAccumulator(std::vector<uint64_t> initialHashes,
                       std::vector<int64_t> partialBlockTokens);

  /**
   * Add a generated token. Returns the updated hash vector if a new block
   * was completed, otherwise nullopt.
   */
  std::optional<std::vector<uint64_t>> addToken(int64_t tokenId);

  const std::vector<uint64_t>& hashes() const { return hashes_; }

 private:
  std::vector<uint64_t> hashes_;
  std::vector<int64_t> blockBuffer_;
  uint64_t parentHash_;
  size_t firstBlockSize_;
  size_t blockSize_;
};

}  // namespace tt::utils
