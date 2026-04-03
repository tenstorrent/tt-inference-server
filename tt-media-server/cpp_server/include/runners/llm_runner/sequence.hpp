#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "runners/llm_runner/sampling_params.hpp"

namespace llm_engine {

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

  static Sequence* deserialize(std::istream& is);

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

  void setKVCacheAddress(uint64_t address) { this->address = address; }

  uint64_t getKVCacheAddress() const { return this->address; }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completionTokenIds() const;

  void appendToken(int64_t tokenId);

  uint32_t taskId;
  SequenceStatus status = SequenceStatus::WAITING;
  std::vector<int64_t> tokenIds;
  int64_t lastToken = 0;
  size_t numPromptTokens = 0;
  size_t numCachedTokens = 0;
  std::vector<int> blockTable;
  std::unique_ptr<SamplingParams> samplingParams;

 private:
  size_t numTokens() const { return tokenIds.size(); }
  int blockSize;
  uint64_t address = 0x0;
};

}  // namespace llm_engine
