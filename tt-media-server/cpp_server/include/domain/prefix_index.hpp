// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <vector>

#include "utils/concurrent_map.hpp"
#include "utils/conversation_hasher.hpp"

namespace tt::domain {

struct Candidate {
  std::string sessionId;
  size_t matchedBlocks;
  size_t sessionBlocks;
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

class PrefixIndex {
 public:
  std::vector<PrefixIndexEntry> getEntriesForKey(uint64_t keyHash) const;
  void registerPrefixHash(const std::string& sessionId,
                          const std::vector<utils::BlockHashInfo>& blockInfos);
  void remove(const std::string& sessionId, uint64_t keyHash);
  void clearThinkTokens(const std::string& sessionId, uint64_t keyHash);

 private:
  utils::ConcurrentMap<uint64_t, std::vector<PrefixIndexEntry>> prefixIndex;
};

}  // namespace tt::domain
