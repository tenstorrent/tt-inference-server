// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <list>
#include <string>

namespace tt::domain::prefix_cache {
struct Candidate {
  std::string sessionId;
  std::size_t matchedBlocks;
  std::size_t sessionBlocks;
  uint32_t thinkTokens;
};

struct RemainingBlockInfo {
  uint64_t hash;
  uint32_t accumulatedThinkTokens;
};

struct PrefixIndexEntry {
  std::list<std::string> sessionIds;
  std::list<RemainingBlockInfo> remainingBlocks;
  uint32_t keyBlockThinkTokens = 0;
};

struct MatchedTokens {
  std::size_t matchedBlocks = 0;
  uint32_t matchedThinkTokens = 0;
};
}  // namespace tt::domain::prefix_cache