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
  static Sequence deserialize(std::istream& is);

  uint32_t taskId;  // set at construction; not const due to IPC deserialization

  size_t size() const { return tokenIds_.size(); }
  int64_t operator[](size_t i) const { return tokenIds_[i]; }

  bool isFinished() const { return status_ == SequenceStatus::FINISHED; }
  bool isAborted() const { return status_ == SequenceStatus::ABORTED; }
  size_t numCompletionTokens() const {
    return tokenIds_.size() - numPromptTokens_;
  }
  size_t numCachedBlocks() const { return numCachedTokens_ / blockSize_; }
  size_t numBlocks() const {
    return (tokenIds_.size() + blockSize_ - 1) / blockSize_;
  }
  int lastBlockNumTokens() const {
    return static_cast<int>(tokenIds_.size()) -
           static_cast<int>(numBlocks() - 1) * blockSize_;
  }

  void setKVCacheAddress(uint64_t addr) { address_ = addr; }
  uint64_t getKVCacheAddress() const { return address_; }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completionTokenIds() const;
  void appendToken(int64_t tokenId);

  SequenceStatus status() const { return status_; }
  void setStatus(SequenceStatus s) { status_ = s; }

  const std::vector<int64_t>& tokenIds() const { return tokenIds_; }

  int64_t lastToken() const { return lastToken_; }
  void setLastToken(int64_t t) { lastToken_ = t; }

  size_t numPromptTokens() const { return numPromptTokens_; }
  void setNumPromptTokens(size_t n) { numPromptTokens_ = n; }

  size_t numCachedTokens() const { return numCachedTokens_; }
  void setNumCachedTokens(size_t n) { numCachedTokens_ = n; }

  const std::vector<int>& blockTable() const { return blockTable_; }
  std::vector<int>& mutableBlockTable() { return blockTable_; }

  const SamplingParams& samplingParams() const { return *samplingParams_; }
  SamplingParams& mutableSamplingParams() { return *samplingParams_; }
  void setSamplingParams(std::unique_ptr<SamplingParams> p) {
    samplingParams_ = std::move(p);
  }

 private:
  SequenceStatus status_ = SequenceStatus::WAITING;
  std::vector<int64_t> tokenIds_;
  int64_t lastToken_ = 0;
  size_t numPromptTokens_ = 0;
  size_t numCachedTokens_ = 0;
  std::vector<int> blockTable_;
  std::unique_ptr<SamplingParams> samplingParams_;
  int blockSize_;
  uint64_t address_ = 0x0;
};

}  // namespace llm_engine
