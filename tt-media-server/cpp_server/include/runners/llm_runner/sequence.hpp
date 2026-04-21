#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "domain/slot_types.hpp"
#include "runners/llm_runner/sampling_params.hpp"

namespace tt::runners::llm_engine {

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

  void serialize(std::ostream& os) const;
  static Sequence deserialize(std::istream& is);

  uint32_t taskId;

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

 private:
  SequenceStatus status = SequenceStatus::WAITING;
  std::vector<int64_t> tokenIds;
  int64_t lastToken = 0;
  size_t numPromptTokens = 0;
  size_t numCachedTokens = 0;
  std::vector<int> blockTable;
  std::unique_ptr<SamplingParams> samplingParams;
  int blockSize;
  uint32_t kvCacheSlot = tt::domain::INVALID_SLOT_ID;
  bool continuation = false;   // True if this continues an existing session
  bool disaggregated = false;  // True if this is a disaggregated request
};

}  // namespace tt::runners::llm_engine
