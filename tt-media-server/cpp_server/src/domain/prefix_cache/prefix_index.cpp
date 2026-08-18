// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/prefix_cache/prefix_index.hpp"

#include <algorithm>

#include "domain/prefix_cache/block_matcher.hpp"

namespace tt::domain::prefix_cache {

namespace {

bool remainingHashesMatch(const std::list<RemainingBlockInfo>& a,
                          const std::list<RemainingBlockInfo>& b) {
  if (a.size() != b.size()) return false;
  auto itA = a.begin();
  auto itB = b.begin();
  while (itA != a.end()) {
    if (itA->hash != itB->hash) return false;
    ++itA;
    ++itB;
  }
  return true;
}

}  // namespace

void PrefixIndex::registerPrefixHash(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) return;
  const uint64_t keyHash = blockInfos.front().hash;
  const uint32_t keyThinkCount = blockInfos.front().accumulatedThinkTokens;
  const std::list<RemainingBlockInfo> remaining =
      BlockMatcher::buildCallerRemaining(blockInfos);

  const bool exists =
      prefixIndex.modify(keyHash, [&sessionId, &remaining, keyThinkCount](
                                      std::vector<PrefixIndexEntry>& entries) {
        for (auto it = entries.begin(); it != entries.end();) {
          it->sessionIds.remove(sessionId);
          if (it->sessionIds.empty()) {
            it = entries.erase(it);
          } else {
            ++it;
          }
        }
        for (auto& entry : entries) {
          if (remainingHashesMatch(entry.remainingBlocks, remaining)) {
            entry.sessionIds.push_back(sessionId);
            return;
          }
        }
        entries.push_back(
            PrefixIndexEntry{{sessionId}, remaining, keyThinkCount});
      });

  if (!exists) {
    std::vector<PrefixIndexEntry> entries;
    entries.push_back(PrefixIndexEntry{{sessionId}, remaining, keyThinkCount});
    prefixIndex.insert(keyHash, std::move(entries));
  }
}

void PrefixIndex::remove(const std::string& sessionId, uint64_t keyHash) {
  if (keyHash == 0) return;

  bool becameEmpty = false;
  prefixIndex.modify(keyHash, [&sessionId, &becameEmpty](
                                  std::vector<PrefixIndexEntry>& entries) {
    for (auto& entry : entries) {
      auto& ids = entry.sessionIds;
      ids.erase(std::remove(ids.begin(), ids.end(), sessionId), ids.end());
    }
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [](const PrefixIndexEntry& e) {
                                   return e.sessionIds.empty();
                                 }),
                  entries.end());
    becameEmpty = entries.empty();
  });
  if (becameEmpty) {
    prefixIndex.erase(keyHash);
  }
}

void PrefixIndex::clearThinkTokens(const std::string& sessionId,
                                   uint64_t keyHash) {
  if (keyHash == 0) return;
  prefixIndex.modify(
      keyHash, [&sessionId](std::vector<PrefixIndexEntry>& entries) {
        for (auto& entry : entries) {
          const bool hasSession =
              std::find(entry.sessionIds.begin(), entry.sessionIds.end(),
                        sessionId) != entry.sessionIds.end();
          if (!hasSession) continue;
          entry.keyBlockThinkTokens = 0;
          for (auto& block : entry.remainingBlocks) {
            block.accumulatedThinkTokens = 0;
          }
        }
      });
}

}  // namespace tt::domain::prefix_cache
