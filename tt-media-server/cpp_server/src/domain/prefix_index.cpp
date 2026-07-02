// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/prefix_index.hpp"

#include <algorithm>

namespace tt::domain {

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

std::list<RemainingBlockInfo> buildRemainingBlocks(
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  std::list<RemainingBlockInfo> remaining;
  for (size_t i = 1; i < blockInfos.size(); ++i) {
    remaining.push_back(
        {blockInfos[i].hash, blockInfos[i].accumulatedThinkTokens});
  }
  return remaining;
}
}  // namespace

std::vector<Candidate> PrefixIndex::findCandidates(
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  std::vector<Candidate> candidates;

  if (blockInfos.empty()) return candidates;

  const uint64_t keyHash = blockInfos.front().hash;

  const std::list<RemainingBlockInfo> callerRemaining =
      buildRemainingBlocks(blockInfos);

  prefixIndex.modify(keyHash, [&](std::vector<PrefixIndexEntry>& entries) {
    for (const auto& entry : entries) {
      size_t matched = 0;

      uint32_t lastMatchedThinkCount = entry.keyBlockThinkTokens;
      auto callerIt = callerRemaining.begin();
      auto entryIt = entry.remainingBlocks.begin();
      while (callerIt != callerRemaining.end() &&
             entryIt != entry.remainingBlocks.end() &&
             callerIt->hash == entryIt->hash) {
        lastMatchedThinkCount = entryIt->accumulatedThinkTokens;
        ++matched;
        ++callerIt;
        ++entryIt;
      }
      const size_t totalMatched = 1 + matched;
      const size_t sessionTotal = 1 + entry.remainingBlocks.size();
      for (const auto& sid : entry.sessionIds) {
        candidates.push_back(
            {sid, totalMatched, sessionTotal, lastMatchedThinkCount});
      }
    }
  });

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.matchedBlocks > b.matchedBlocks;
            });
  return candidates;
}

void PrefixIndex::registerPrefixHash(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) return;
  const uint64_t keyHash = blockInfos.front().hash;
  const uint32_t keyThinkCount = blockInfos.front().accumulatedThinkTokens;
  const std::list<RemainingBlockInfo> remaining =
      buildRemainingBlocks(blockInfos);
  bool exists =
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
std::pair<size_t, uint32_t> PrefixIndex::computeMatchedBlocks(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) {
    return {0, 0};
  }
  const uint64_t keyHash = blockInfos.front().hash;
  const std::list<RemainingBlockInfo> callerRemaining =
      buildRemainingBlocks(blockInfos);
  size_t matchedBlocks = 0;
  uint32_t thinkTokens = 0;
  prefixIndex.modify(keyHash, [&](std::vector<PrefixIndexEntry>& entries) {
    for (const auto& entry : entries) {
      const bool hasSession =
          std::find(entry.sessionIds.begin(), entry.sessionIds.end(),
                    sessionId) != entry.sessionIds.end();
      if (!hasSession) continue;
      size_t matched = 0;
      uint32_t lastThink = entry.keyBlockThinkTokens;
      auto callerIt = callerRemaining.begin();
      auto entryIt = entry.remainingBlocks.begin();
      while (callerIt != callerRemaining.end() &&
             entryIt != entry.remainingBlocks.end() &&
             callerIt->hash == entryIt->hash) {
        lastThink = entryIt->accumulatedThinkTokens;
        ++matched;
        ++callerIt;
        ++entryIt;
      }
      const size_t total = 1 + matched;
      if (total > matchedBlocks) {
        matchedBlocks = total;
        thinkTokens = lastThink;
      }
    }
  });
  return {matchedBlocks, thinkTokens};
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

}  // namespace tt::domain