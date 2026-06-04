// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "domain/llm/sampling_params.hpp"
#include "domain/sentinel_values.hpp"

namespace tt::domain::llm {

enum class SequenceStatus { WAITING, RUNNING, IN_FLIGHT, FINISHED, ABORTED };

struct TokenResult {
  uint32_t taskId;
  uint64_t tokenId = 0;
  std::optional<bool> finished;
  bool isError = false;

  TokenResult() = default;
  TokenResult(uint32_t taskId, uint64_t tokenId,
              std::optional<bool> finished = {}, bool isError = false)
      : taskId(taskId),
        tokenId(tokenId),
        finished(std::move(finished)),
        isError(isError) {}
};

class Sequence {
 public:
  Sequence(uint32_t taskId, int blockSize, std::vector<int64_t> tokenIds,
           const SamplingParams& samplingParams = SamplingParams());

  Sequence(uint32_t taskId, int blockSize, std::vector<int64_t> tokenIds,
           size_t numPromptTokens, std::optional<uint32_t> slotId,
           std::optional<uint32_t> prefillSlotId, bool continuation,
           bool disaggregated, std::unique_ptr<SamplingParams> samplingParams,
           std::optional<uint32_t> kvPositionId = std::nullopt,
           int numberOfDecodeSkipTokens = 0, std::string traceId = "");

  void serialize(std::ostream& os) const;
  static Sequence deserialize(std::istream& is);

  uint32_t taskId;
  // End-to-end trace id (issue #3929): set on the HTTP node, carried through
  // taskQueue serialization so worker logs can grep the same id as the
  // controller/socket layer logs.
  std::string traceId;

  size_t size() const { return tokenIds.size(); }
  int64_t operator[](size_t i) const { return tokenIds[i]; }

  bool isFinished() const { return status == SequenceStatus::FINISHED; }
  bool isAborted() const { return status == SequenceStatus::ABORTED; }
  size_t numCompletionTokens() const {
    return tokenIds.size() - numPromptTokens;
  }
  size_t numCachedBlocks() const { return numCachedTokens / blockSize; }
  size_t numBlocks() const {
    return (tokenIds.size() + blockSize - 1) / blockSize;
  }
  int lastBlockNumTokens() const {
    return static_cast<int>(tokenIds.size()) -
           static_cast<int>(numBlocks() - 1) * blockSize;
  }

  void setKVCacheSlot(uint32_t slot) { kvCacheSlot = slot; }
  uint32_t getKVCacheSlot() const { return kvCacheSlot; }

  void setPrefillKVCacheSlot(uint32_t slot) { prefillKvCacheSlot = slot; }
  uint32_t getPrefillKVCacheSlot() const { return prefillKvCacheSlot; }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completionTokenIds() const;
  void appendToken(int64_t tokenId);

  SequenceStatus getStatus() const { return status; }
  void setStatus(SequenceStatus s) { status = s; }

  const std::vector<int64_t>& getTokenIds() const { return tokenIds; }

  int64_t getLastToken() const { return lastToken; }
  void setLastToken(int64_t t) { lastToken = t; }

  size_t getNumPromptTokens() const { return numPromptTokens; }
  void setNumPromptTokens(size_t n) { numPromptTokens = n; }

  size_t getNumCachedTokens() const { return numCachedTokens; }
  void setNumCachedTokens(size_t n) { numCachedTokens = n; }

  const std::vector<int>& getBlockTable() const { return blockTable; }
  std::vector<int>& getMutableBlockTable() { return blockTable; }

  const SamplingParams& getSamplingParams() const { return *samplingParams; }
  SamplingParams& getMutableSamplingParams() { return *samplingParams; }
  void setSamplingParams(std::unique_ptr<SamplingParams> p) {
    samplingParams = std::move(p);
  }

  bool isContinuation() const { return continuation; }
  void setContinuation(bool c) { continuation = c; }

  bool isDisaggregated() const { return disaggregated; }
  void setDisaggregated(bool d) { disaggregated = d; }

  std::optional<uint32_t> getKVPositionId() const { return kvPositionId; }
  void setKVPositionId(uint32_t positionId) { kvPositionId = positionId; }

  int getNumberOfDecodeSkipTokens() const { return numberOfDecodeSkipTokens; }
  void setNumberOfDecodeSkipTokens(int n) { numberOfDecodeSkipTokens = n; }

 private:
  SequenceStatus status = SequenceStatus::WAITING;
  std::vector<int64_t> tokenIds;
  int64_t lastToken = 0;
  std::optional<uint32_t> kvPositionId = std::nullopt;
  size_t numPromptTokens = 0;
  size_t numCachedTokens = 0;
  std::vector<int> blockTable;
  std::unique_ptr<SamplingParams> samplingParams;
  int blockSize;
  uint32_t kvCacheSlot = tt::domain::INVALID_SLOT_ID;
  uint32_t prefillKvCacheSlot = tt::domain::INVALID_SLOT_ID;
  bool continuation = false;   // True if this continues an existing session
  bool disaggregated = false;  // True if this is a disaggregated request
  int numberOfDecodeSkipTokens = 0;
};

}  // namespace tt::domain::llm
