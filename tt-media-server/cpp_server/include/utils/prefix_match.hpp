// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>

namespace tt::utils {

template <typename HashRange, typename ContainsHash>
size_t countMatchingPrefixDepth(const HashRange& hashes,
                                ContainsHash containsHash) {
  size_t depth = 0;
  for (const auto& hash : hashes) {
    if (!containsHash(hash)) {
      break;
    }
    ++depth;
  }
  return depth;
}

template <typename LeftIt, typename RightIt, typename LeftHash,
          typename RightHash>
size_t countEqualPrefix(LeftIt left, LeftIt leftEnd, RightIt right,
                        RightIt rightEnd, LeftHash leftHash,
                        RightHash rightHash) {
  size_t depth = 0;
  while (left != leftEnd && right != rightEnd &&
         leftHash(*left) == rightHash(*right)) {
    ++depth;
    ++left;
    ++right;
  }
  return depth;
}

}  // namespace tt::utils
