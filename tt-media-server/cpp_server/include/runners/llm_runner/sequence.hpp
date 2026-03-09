#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "runners/llm_runner/sampling_params.hpp"
#include "domain/task_id.hpp"

namespace llm_engine {

using TaskID = tt::domain::TaskID;

enum class SequenceStatus { WAITING, RUNNING, IN_FLIGHT, FINISHED };

struct TokenResult {
  TaskID task_id;
  uint64_t token_id;
  std::optional<bool> finished;
  bool is_error = false;
};

class Sequence {
 public:
  Sequence(const int block_size, std::vector<int64_t> token_ids,
           const SamplingParams& sampling_params = SamplingParams());

  void serialize(std::ostream& os) const;

  static Sequence* deserialize(std::istream& is);

  size_t size() const { return token_ids_.size(); }
  int64_t operator[](size_t i) const { return token_ids_[i]; }

  bool is_finished() const { return status_ == SequenceStatus::FINISHED; }
  size_t num_completion_tokens() const {
    return token_ids_.size() - num_prompt_tokens_;
  }
  size_t num_cached_blocks() const { return num_cached_tokens_ / block_size_; }
  size_t num_blocks() const {
    return (token_ids_.size() + block_size_ - 1) / block_size_;
  }
  int last_block_num_tokens() const {
    return static_cast<int>(token_ids_.size()) -
           static_cast<int>(num_blocks() - 1) * block_size_;
  }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completion_token_ids() const;

  void append_token(int64_t token_id);

  TaskID task_id;
  SequenceStatus status_ = SequenceStatus::WAITING;
  std::vector<int64_t> token_ids_;
  int64_t last_token = 0;
  size_t num_prompt_tokens_ = 0;
  size_t num_cached_tokens_ = 0;
  std::vector<int> block_table_;
  std::unique_ptr<SamplingParams> sampling_params;

 private:
  size_t num_tokens() const { return token_ids_.size(); }
  int block_size_;
};

}  // namespace llm_engine
