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

// Note: status_ is intentionally not serialized. Sequences in the queue are
// always in WAITING status (the Sequence constructor default).
void Sequence::serialize(std::ostream& os) const {
  size_t token_ids_size = token_ids_.size();
  size_t block_table_size = block_table_.size();
  os.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
  os.write(reinterpret_cast<const char*>(&last_token), sizeof(last_token));
  os.write(reinterpret_cast<const char*>(&num_prompt_tokens_), sizeof(num_prompt_tokens_));
  os.write(reinterpret_cast<const char*>(&num_cached_tokens_), sizeof(num_cached_tokens_));
  os.write(reinterpret_cast<const char*>(&max_tokens), sizeof(max_tokens));
  os.write(reinterpret_cast<const char*>(&ignore_eos), sizeof(ignore_eos));
  os.write(reinterpret_cast<const char*>(&token_ids_size), sizeof(token_ids_size));
  os.write(reinterpret_cast<const char*>(token_ids_.data()), token_ids_size * sizeof(int64_t));
  os.write(reinterpret_cast<const char*>(&block_table_size), sizeof(block_table_size));
  os.write(reinterpret_cast<const char*>(block_table_.data()), block_table_size * sizeof(int));
  os.write(reinterpret_cast<const char*>(&temperature), sizeof(temperature));
  os.write(reinterpret_cast<const char*>(&status_), sizeof(status_));
}

Sequence* Sequence::deserialize(std::istream& is) {
  Sequence* seq = new Sequence(std::vector<int64_t>{});

  is.read(reinterpret_cast<char*>(&seq->seq_id), sizeof(seq->seq_id));
  is.read(reinterpret_cast<char*>(&seq->last_token), sizeof(seq->last_token));
  is.read(reinterpret_cast<char*>(&seq->num_prompt_tokens_), sizeof(seq->num_prompt_tokens_));
  is.read(reinterpret_cast<char*>(&seq->num_cached_tokens_), sizeof(seq->num_cached_tokens_));
  is.read(reinterpret_cast<char*>(&seq->max_tokens), sizeof(seq->max_tokens));
  is.read(reinterpret_cast<char*>(&seq->ignore_eos), sizeof(seq->ignore_eos));

  size_t token_ids_size;
  is.read(reinterpret_cast<char*>(&token_ids_size), sizeof(token_ids_size));
  seq->token_ids_.resize(token_ids_size);
  is.read(reinterpret_cast<char*>(seq->token_ids_.data()), token_ids_size * sizeof(int64_t));

  size_t block_table_size;
  is.read(reinterpret_cast<char*>(&block_table_size), sizeof(block_table_size));
  seq->block_table_.resize(block_table_size);
  is.read(reinterpret_cast<char*>(seq->block_table_.data()), block_table_size * sizeof(int));

  is.read(reinterpret_cast<char*>(&seq->temperature), sizeof(seq->temperature));
  
  is.read(reinterpret_cast<char*>(&seq->status_), sizeof(seq->status_));
  return seq;
}

}  // namespace llm_engine
