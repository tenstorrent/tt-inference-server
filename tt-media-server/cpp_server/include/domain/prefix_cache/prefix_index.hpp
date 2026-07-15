// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <vector>

#include "helpers.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/conversation_hasher.hpp"

namespace tt::domain::prefix_cache {
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

}  // namespace tt::domain::prefix_cache
