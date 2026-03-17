#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "domain/task_id.hpp"
#include "runners/llm_runner/sampling_params.hpp"

namespace llm_engine {

using TaskID = tt::domain::TaskID;

enum class SequenceStatus { WAITING, RUNNING, IN_FLIGHT, FINISHED };

struct TokenResult {
  TaskID task_id;
  uint64_t token_id = 0;
  std::optional<bool> finished;
  bool is_error = false;

  TokenResult() = default;
  TokenResult(TaskID taskId, uint64_t tokenId,
              std::optional<bool> finished = {}, bool isError = false)
      : task_id(std::move(taskId)),
        token_id(tokenId),
        finished(std::move(finished)),
        is_error(isError) {}
};

class Sequence {
 public:
  Sequence(TaskID taskId, int blockSize, std::vector<int64_t> tokenIds,
           const SamplingParams& samplingParams = SamplingParams());

  void serialize(std::ostream& os) const;

  static Sequence* deserialize(std::istream& is);

  size_t size() const { return token_ids_.size(); }
  int64_t operator[](size_t i) const { return token_ids_[i]; }

  bool isFinished() const { return status_ == SequenceStatus::FINISHED; }
  size_t numCompletionTokens() const {
    return token_ids_.size() - num_prompt_tokens_;
  }
  size_t numCachedBlocks() const { return num_cached_tokens_ / block_size_; }
  size_t numBlocks() const {
    return (token_ids_.size() + block_size_ - 1) / block_size_;
  }
  int lastBlockNumTokens() const {
    return static_cast<int>(token_ids_.size()) -
           static_cast<int>(numBlocks() - 1) * block_size_;
  }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completionTokenIds() const;

  void appendToken(int64_t tokenId);

  TaskID task_id;
  SequenceStatus status_ = SequenceStatus::WAITING;
  std::vector<int64_t> token_ids_;
  int64_t last_token = 0;
  size_t num_prompt_tokens_ = 0;
  size_t num_cached_tokens_ = 0;
  std::vector<int> block_table_;
  std::unique_ptr<SamplingParams> sampling_params;

 private:
  size_t numTokens() const { return token_ids_.size(); }
  int block_size_;
};

}  // namespace llm_engine
