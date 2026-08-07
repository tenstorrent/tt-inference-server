// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "domain/llm/sequence.hpp"

#include <cstdint>
#include <utility>

#include "domain/llm/sampling_params.hpp"

namespace tt::domain::llm {

Sequence::Sequence(uint32_t taskId, std::vector<uint32_t> inputTokenIds,
                   const SamplingParams& inputSamplingParams)
    : taskId(taskId),
      tokenIds(std::move(inputTokenIds)),
      samplingParams(std::make_unique<SamplingParams>(inputSamplingParams)),
      numPromptTokens(tokenIds.size()) {}

Sequence::Sequence(uint32_t taskId, std::vector<uint32_t> inputTokenIds,
                   size_t numPromptTokens, std::optional<uint32_t> slotId,
                   std::optional<uint32_t> prefillSlotId, bool continuation,
                   bool disaggregated,
                   std::unique_ptr<SamplingParams> inputSamplingParams,
                   std::optional<uint32_t> kvPositionId, int decodePositionId,
                   int decodeSkipTokens, std::optional<uint64_t> migrationId,
                   bool startsInThinking,
                   std::optional<uint32_t> migrationStartPosition)
    : taskId(taskId),
      tokenIds(std::move(inputTokenIds)),
      samplingParams(std::move(inputSamplingParams)),
      numPromptTokens(numPromptTokens),
      kvCacheSlot(slotId.value_or(tt::domain::INVALID_SLOT_ID)),
      prefillKvCacheSlot(prefillSlotId.value_or(tt::domain::INVALID_SLOT_ID)),
      kvPositionId(std::move(kvPositionId)),
      decodePositionId(decodePositionId),
      decodeSkipTokens(decodeSkipTokens),
      migrationId(migrationId),
      migrationStartPosition(std::move(migrationStartPosition)),
      continuation(continuation),
      disaggregated(disaggregated),
      startsInThinking_(startsInThinking) {}

void Sequence::serialize(std::ostream& os) const {
  size_t tokenIdsSize = tokenIds.size();
  os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
  os.write(reinterpret_cast<const char*>(&numPromptTokens),
           sizeof(numPromptTokens));
  os.write(reinterpret_cast<const char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  os.write(reinterpret_cast<const char*>(tokenIds.data()),
           tokenIdsSize * sizeof(uint32_t));
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
    uint32_t migrationStartPositionValue = migrationStartPosition.value();
    os.write(reinterpret_cast<const char*>(&migrationStartPositionValue),
             sizeof(migrationStartPositionValue));
  }
}

Sequence Sequence::deserialize(std::istream& is) {
  uint32_t taskId;
  is.read(reinterpret_cast<char*>(&taskId), sizeof(taskId));

  Sequence seq(taskId, std::vector<uint32_t>{});

  is.read(reinterpret_cast<char*>(&seq.numPromptTokens),
          sizeof(seq.numPromptTokens));

  size_t tokenIdsSize;
  is.read(reinterpret_cast<char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  seq.tokenIds.resize(tokenIdsSize);
  is.read(reinterpret_cast<char*>(seq.tokenIds.data()),
          tokenIdsSize * sizeof(uint32_t));

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
  uint8_t hasKvPositionId = 0;
  is.read(reinterpret_cast<char*>(&hasKvPositionId), sizeof(hasKvPositionId));
  if (hasKvPositionId) {
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
    seq.migrationStartPosition = std::make_optional<uint32_t>(0);
    is.read(reinterpret_cast<char*>(&(*seq.migrationStartPosition)),
            sizeof(*seq.migrationStartPosition));
  } else {
    seq.migrationStartPosition = std::nullopt;
  }
  return seq;
}
}  // namespace tt::domain::llm
