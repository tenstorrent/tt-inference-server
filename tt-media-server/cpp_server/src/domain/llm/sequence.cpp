// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "domain/llm/sequence.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>

#include "config/runner_config.hpp"
#include "domain/llm/sampling_params.hpp"

namespace tt::domain::llm {

using Config = tt::config::LLMConfig;

Sequence::Sequence(uint32_t taskId, int blockSize,
                   std::vector<int64_t> inputTokenIds,
                   const SamplingParams& inputSamplingParams)
    : taskId(taskId),
      status(SequenceStatus::WAITING),
      tokenIds(std::move(inputTokenIds)),
      numPromptTokens(tokenIds.size()),
      samplingParams(std::make_unique<SamplingParams>(inputSamplingParams)),
      blockSize(blockSize) {
  if (!tokenIds.empty()) {
    lastToken = tokenIds.back();
  }
}

Sequence::Sequence(uint32_t taskId, int blockSize,
                   std::vector<int64_t> inputTokenIds, size_t numPromptTokens,
                   std::optional<uint32_t> slotId,
                   std::optional<uint32_t> prefillSlotId, bool continuation,
                   bool disaggregated,
                   std::unique_ptr<SamplingParams> inputSamplingParams,
                   std::optional<uint32_t> kvPositionId, int decodePositionId,
                   int decodeSkipTokens, std::optional<uint64_t> migrationId,
                   bool startsInThinking,
                   std::optional<uint64_t> migrationStartPosition)
    : taskId(taskId),
      status(SequenceStatus::WAITING),
      tokenIds(std::move(inputTokenIds)),
      kvPositionId(std::move(kvPositionId)),
      numPromptTokens(numPromptTokens),
      samplingParams(std::move(inputSamplingParams)),
      blockSize(blockSize),
      kvCacheSlot(slotId.value_or(tt::domain::INVALID_SLOT_ID)),
      prefillKvCacheSlot(prefillSlotId.value_or(tt::domain::INVALID_SLOT_ID)),
      continuation(continuation),
      disaggregated(disaggregated),
      decodePositionId(decodePositionId),
      decodeSkipTokens(decodeSkipTokens),
      migrationId(migrationId),
      migrationStartPosition(std::move(migrationStartPosition)),
      startsInThinking_(startsInThinking) {
  if (!tokenIds.empty()) {
    lastToken = tokenIds.back();
  }
}

std::vector<int64_t> Sequence::block(size_t i) const {
  size_t n = numBlocks();
  if (i >= n) {
    throw std::out_of_range("block index out of range");
  }
  size_t start = i * blockSize;
  size_t end = std::min(start + blockSize, tokenIds.size());
  return {tokenIds.begin() + start, tokenIds.begin() + end};
}

std::vector<int64_t> Sequence::completionTokenIds() const {
  if (numPromptTokens >= tokenIds.size()) {
    return {};
  }
  return {tokenIds.begin() + numPromptTokens, tokenIds.end()};
}

void Sequence::appendToken(int64_t tokenId) {
  tokenIds.push_back(tokenId);
  lastToken = tokenId;
}

void Sequence::serialize(std::ostream& os) const {
  size_t tokenIdsSize = tokenIds.size();
  size_t blockTableSize = blockTable.size();
  os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
  os.write(reinterpret_cast<const char*>(&lastToken), sizeof(lastToken));
  os.write(reinterpret_cast<const char*>(&numPromptTokens),
           sizeof(numPromptTokens));
  os.write(reinterpret_cast<const char*>(&numCachedTokens),
           sizeof(numCachedTokens));
  os.write(reinterpret_cast<const char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  os.write(reinterpret_cast<const char*>(tokenIds.data()),
           tokenIdsSize * sizeof(int64_t));
  os.write(reinterpret_cast<const char*>(&blockTableSize),
           sizeof(blockTableSize));
  os.write(reinterpret_cast<const char*>(blockTable.data()),
           blockTableSize * sizeof(int));
  os.write(reinterpret_cast<const char*>(&status), sizeof(status));
  os.write(reinterpret_cast<const char*>(&blockSize), sizeof(blockSize));
  os.write(reinterpret_cast<const char*>(&kvCacheSlot), sizeof(kvCacheSlot));
  os.write(reinterpret_cast<const char*>(&prefillKvCacheSlot),
           sizeof(prefillKvCacheSlot));
  uint8_t continuationFlag = continuation ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&continuationFlag),
           sizeof(continuationFlag));
  uint8_t disaggregatedFlag = disaggregated ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&disaggregatedFlag),
           sizeof(disaggregatedFlag));
  samplingParams->serialize(os);
  uint8_t hasKvPositionId = kvPositionId.has_value() ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&hasKvPositionId),
           sizeof(hasKvPositionId));
  if (hasKvPositionId) {
    uint32_t kvPositionIdValue = kvPositionId.value();
    os.write(reinterpret_cast<const char*>(&kvPositionIdValue),
             sizeof(uint32_t));
  }
  os.write(reinterpret_cast<const char*>(&decodePositionId),
           sizeof(decodePositionId));
  os.write(reinterpret_cast<const char*>(&decodeSkipTokens),
           sizeof(decodeSkipTokens));
  uint8_t hasMigrationId = migrationId.has_value() ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&hasMigrationId),
           sizeof(hasMigrationId));
  if (hasMigrationId) {
    uint64_t migrationIdValue = migrationId.value();
    os.write(reinterpret_cast<const char*>(&migrationIdValue),
             sizeof(migrationIdValue));
  }
  uint8_t startsInThinkingFlag = startsInThinking_ ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&startsInThinkingFlag),
           sizeof(startsInThinkingFlag));
  uint8_t hasMigrationStartPosition =
      migrationStartPosition.has_value() ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&hasMigrationStartPosition),
           sizeof(hasMigrationStartPosition));
  if (hasMigrationStartPosition) {
    uint64_t migrationStartPositionValue = migrationStartPosition.value();
    os.write(reinterpret_cast<const char*>(&migrationStartPositionValue),
             sizeof(migrationStartPositionValue));
  }
}

Sequence Sequence::deserialize(std::istream& is) {
  uint32_t taskId;
  is.read(reinterpret_cast<char*>(&taskId), sizeof(taskId));

  Config defaultConfig;
  Sequence seq(taskId, static_cast<int>(defaultConfig.kvcache_block_size),
               std::vector<int64_t>{});

  is.read(reinterpret_cast<char*>(&seq.lastToken), sizeof(seq.lastToken));
  is.read(reinterpret_cast<char*>(&seq.numPromptTokens),
          sizeof(seq.numPromptTokens));
  is.read(reinterpret_cast<char*>(&seq.numCachedTokens),
          sizeof(seq.numCachedTokens));

  size_t tokenIdsSize;
  is.read(reinterpret_cast<char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  seq.tokenIds.resize(tokenIdsSize);
  is.read(reinterpret_cast<char*>(seq.tokenIds.data()),
          tokenIdsSize * sizeof(int64_t));

  size_t blockTableSize;
  is.read(reinterpret_cast<char*>(&blockTableSize), sizeof(blockTableSize));
  seq.blockTable.resize(blockTableSize);
  is.read(reinterpret_cast<char*>(seq.blockTable.data()),
          blockTableSize * sizeof(int));

  is.read(reinterpret_cast<char*>(&seq.status), sizeof(seq.status));
  is.read(reinterpret_cast<char*>(&seq.blockSize), sizeof(seq.blockSize));
  is.read(reinterpret_cast<char*>(&seq.kvCacheSlot), sizeof(seq.kvCacheSlot));
  is.read(reinterpret_cast<char*>(&seq.prefillKvCacheSlot),
          sizeof(seq.prefillKvCacheSlot));
  uint8_t continuationFlag = 0;
  is.read(reinterpret_cast<char*>(&continuationFlag), sizeof(continuationFlag));
  seq.continuation = continuationFlag != 0;
  uint8_t disaggregatedFlag = 0;
  is.read(reinterpret_cast<char*>(&disaggregatedFlag),
          sizeof(disaggregatedFlag));
  seq.disaggregated = disaggregatedFlag != 0;
  seq.samplingParams = SamplingParams::deserialize(is);
  uint8_t hasKvCacheOffset = 0;
  is.read(reinterpret_cast<char*>(&hasKvCacheOffset), sizeof(hasKvCacheOffset));
  if (hasKvCacheOffset) {
    seq.kvPositionId = std::make_optional<uint32_t>(0);
    is.read(reinterpret_cast<char*>(&(*seq.kvPositionId)),
            sizeof(*seq.kvPositionId));
  } else {
    seq.kvPositionId = std::nullopt;
  }
  is.read(reinterpret_cast<char*>(&seq.decodePositionId),
          sizeof(seq.decodePositionId));
  is.read(reinterpret_cast<char*>(&seq.decodeSkipTokens),
          sizeof(seq.decodeSkipTokens));
  uint8_t hasMigrationId = 0;
  is.read(reinterpret_cast<char*>(&hasMigrationId), sizeof(hasMigrationId));
  if (hasMigrationId) {
    seq.migrationId = std::make_optional<uint64_t>(0);
    is.read(reinterpret_cast<char*>(&(*seq.migrationId)),
            sizeof(*seq.migrationId));
  } else {
    seq.migrationId = std::nullopt;
  }
  uint8_t startsInThinkingFlag = 0;
  is.read(reinterpret_cast<char*>(&startsInThinkingFlag),
          sizeof(startsInThinkingFlag));
  seq.startsInThinking_ = startsInThinkingFlag != 0;
  uint8_t hasMigrationStartPosition = 0;
  is.read(reinterpret_cast<char*>(&hasMigrationStartPosition),
          sizeof(hasMigrationStartPosition));
  if (hasMigrationStartPosition) {
    seq.migrationStartPosition = std::make_optional<uint64_t>(0);
    is.read(reinterpret_cast<char*>(&(*seq.migrationStartPosition)),
            sizeof(*seq.migrationStartPosition));
  } else {
    seq.migrationStartPosition = std::nullopt;
  }
  return seq;
}
}  // namespace tt::domain::llm
