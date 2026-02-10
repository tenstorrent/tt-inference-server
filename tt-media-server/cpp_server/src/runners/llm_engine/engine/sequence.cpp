#include "llm_engine/engine/sequence.hpp"
#include <algorithm>
#include <stdexcept>

namespace llm_engine {

int Sequence::next_seq_id() {
  static int counter = 0;
  return counter++;
}

Sequence::Sequence(std::vector<int64_t> token_ids,
                   const SamplingParams& sampling_params)
    : seq_id(next_seq_id()),
      status_(SequenceStatus::WAITING),
      token_ids_(std::move(token_ids)),
      num_prompt_tokens_(token_ids_.size()),
      temperature(sampling_params.temperature),
      max_tokens(sampling_params.max_tokens),
      ignore_eos(sampling_params.ignore_eos) {
  if (!token_ids_.empty()) {
    last_token = token_ids_.back();
  }
}

Sequence::Sequence(int seq_id, std::vector<int64_t> token_ids,
                   size_t num_prompt_tokens, size_t num_cached_tokens,
                   float temperature, int max_tokens, bool ignore_eos)
    : seq_id(seq_id),
      status_(SequenceStatus::WAITING),
      token_ids_(std::move(token_ids)),
      num_prompt_tokens_(num_prompt_tokens),
      num_cached_tokens_(num_cached_tokens),
      temperature(temperature),
      max_tokens(max_tokens),
      ignore_eos(ignore_eos) {
  if (!token_ids_.empty()) {
    last_token = token_ids_.back();
  }
}

std::vector<int64_t> Sequence::block(size_t i) const {
  size_t n = num_blocks();
  if (i >= n) {
    throw std::out_of_range("block index out of range");
  }
  size_t start = i * block_size;
  size_t end = std::min(start + block_size, token_ids_.size());
  return std::vector<int64_t>(token_ids_.begin() + start, token_ids_.begin() + end);
}

std::vector<int64_t> Sequence::completion_token_ids() const {
  if (num_prompt_tokens_ >= token_ids_.size()) {
    return {};
  }
  return std::vector<int64_t>(token_ids_.begin() + num_prompt_tokens_,
                              token_ids_.end());
}

void Sequence::append_token(int64_t token_id) {
  token_ids_.push_back(token_id);
  last_token = token_id;
}

}  // namespace llm_engine
