// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "domain/llm/sampling_params.hpp"
#include "domain/sentinel_values.hpp"

namespace tt::domain::llm {

struct TokenResult {
  uint32_t taskId;
  uint32_t tokenId = 0;
  std::optional<bool> finished;
  bool isError = false;

  TokenResult() = default;
  TokenResult(uint32_t taskId, uint32_t tokenId,
              std::optional<bool> finished = {}, bool isError = false)
      : taskId(taskId),
        tokenId(tokenId),
        finished(std::move(finished)),
        isError(isError) {}
};

// Inference request carried across the HTTP node -> worker task queue. The
// binary serialize()/deserialize() format is the IPC wire format, so both ends
// must stay in lockstep (they are the same binary).
class Sequence {
 public:
  Sequence(uint32_t taskId, std::vector<uint32_t> tokenIds,
           const SamplingParams& samplingParams = SamplingParams());

  Sequence(uint32_t taskId, std::vector<uint32_t> tokenIds,
           size_t numPromptTokens, std::optional<uint32_t> slotId,
           std::optional<uint32_t> prefillSlotId, bool continuation,
           bool disaggregated, std::unique_ptr<SamplingParams> samplingParams,
           std::optional<uint32_t> kvPositionId = std::nullopt,
           int decodePositionId = 0, int decodeSkipTokens = 0,
           std::optional<uint64_t> migrationId = std::nullopt,
           bool startsInThinking = false,
           std::optional<uint32_t> migrationStartPosition = std::nullopt);

  void serialize(std::ostream& os) const;
  static Sequence deserialize(std::istream& is);

  uint32_t taskId;

  const std::vector<uint32_t>& getTokenIds() const { return tokenIds; }

  size_t getNumPromptTokens() const { return numPromptTokens; }

  void setKVCacheSlot(uint32_t slot) { kvCacheSlot = slot; }
  uint32_t getKVCacheSlot() const { return kvCacheSlot; }

  void setPrefillKVCacheSlot(uint32_t slot) { prefillKvCacheSlot = slot; }
  uint32_t getPrefillKVCacheSlot() const { return prefillKvCacheSlot; }

  const SamplingParams& getSamplingParams() const { return *samplingParams; }

  bool isContinuation() const { return continuation; }
  bool isDisaggregated() const { return disaggregated; }

  std::optional<uint32_t> getKVPositionId() const { return kvPositionId; }
  int getDecodePositionId() const { return decodePositionId; }
  int getDecodeSkipTokens() const { return decodeSkipTokens; }

  std::optional<uint64_t> getMigrationId() const { return migrationId; }
  std::optional<uint32_t> getMigrationStartPosition() const {
    return migrationStartPosition;
  }

  bool getStartsInThinking() const { return startsInThinking_; }

 private:
  std::vector<uint32_t> tokenIds;
  std::unique_ptr<SamplingParams> samplingParams;
  size_t numPromptTokens = 0;
  uint32_t kvCacheSlot = tt::domain::INVALID_SLOT_ID;
  uint32_t prefillKvCacheSlot = tt::domain::INVALID_SLOT_ID;
  std::optional<uint32_t> kvPositionId = std::nullopt;
  int decodePositionId = 0;
  int decodeSkipTokens = 0;
  std::optional<uint64_t> migrationId;
  std::optional<uint32_t> migrationStartPosition;
  bool continuation = false;
  bool disaggregated = false;
  bool startsInThinking_ = false;
};

}  // namespace tt::domain::llm
