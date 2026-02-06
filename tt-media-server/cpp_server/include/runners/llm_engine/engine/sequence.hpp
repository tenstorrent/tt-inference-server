#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llm_engine/sampling_params.hpp"

namespace llm_engine {

enum class SequenceStatus { WAITING, RUNNING, FINISHED };

class Sequence {
 public:
  static constexpr int block_size = 256;
  static int next_seq_id();

  Sequence(std::vector<int64_t> token_ids,
           const SamplingParams& sampling_params = SamplingParams());

  size_t size() const { return token_ids_.size(); }
  int64_t operator[](size_t i) const { return token_ids_[i]; }

  bool is_finished() const { return status_ == SequenceStatus::FINISHED; }
  size_t num_completion_tokens() const {
    return token_ids_.size() - num_prompt_tokens_;
  }
  size_t num_cached_blocks() const { return num_cached_tokens_ / block_size; }
  size_t num_blocks() const {
    return (token_ids_.size() + block_size - 1) / block_size;
  }
  int last_block_num_tokens() const {
    return static_cast<int>(token_ids_.size()) -
           static_cast<int>(num_blocks() - 1) * block_size;
  }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completion_token_ids() const;

  void append_token(int64_t token_id);

  int seq_id = 0;
  SequenceStatus status_ = SequenceStatus::WAITING;
  std::vector<int64_t> token_ids_;
  int64_t last_token = 0;
  size_t num_prompt_tokens_ = 0;
  size_t num_cached_tokens_ = 0;
  std::vector<int> block_table_;
  float temperature = 1.0f;
  /** Max completion tokens for this sequence (from SamplingParams). Each request can have a different value. */
  int max_tokens = 64;
  bool ignore_eos = false;

 private:
  size_t num_tokens() const { return token_ids_.size(); }
};

}  // namespace llm_engine
