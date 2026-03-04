#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/config.hpp"
#include "llm_runner/sampling_params.hpp"
#include <algorithm>
#include <stdexcept>

namespace llm_engine {

Sequence::Sequence(const int block_size, std::vector<int64_t> token_ids,
                   const SamplingParams& sampling_params)
    : task_id(TaskID{}),
      status_(SequenceStatus::WAITING),
      token_ids_(std::move(token_ids)),
      num_prompt_tokens_(token_ids_.size()),
      sampling_params(std::make_unique<SamplingParams>(sampling_params)),
      block_size_(block_size) {
  if (!token_ids_.empty()) {
    last_token = token_ids_.back();
  }
}

std::vector<int64_t> Sequence::block(size_t i) const {
  size_t n = num_blocks();
  if (i >= n) {
    throw std::out_of_range("block index out of range");
  }
  size_t start = i * block_size_;
  size_t end = std::min(start + block_size_, token_ids_.size());
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

void Sequence::serialize(std::ostream& os) const {
  size_t token_ids_size = token_ids_.size();
  size_t block_table_size = block_table_.size();
  size_t id_size = task_id.id.size();
  os.write(reinterpret_cast<const char*>(&id_size), sizeof(id_size));
  os.write(task_id.id.c_str(), id_size);
  os.write(reinterpret_cast<const char*>(&last_token), sizeof(last_token));
  os.write(reinterpret_cast<const char*>(&num_prompt_tokens_), sizeof(num_prompt_tokens_));
  os.write(reinterpret_cast<const char*>(&num_cached_tokens_), sizeof(num_cached_tokens_));
  os.write(reinterpret_cast<const char*>(&token_ids_size), sizeof(token_ids_size));
  os.write(reinterpret_cast<const char*>(token_ids_.data()), token_ids_size * sizeof(int64_t));
  os.write(reinterpret_cast<const char*>(&block_table_size), sizeof(block_table_size));
  os.write(reinterpret_cast<const char*>(block_table_.data()), block_table_size * sizeof(int));
  os.write(reinterpret_cast<const char*>(&status_), sizeof(status_));
  os.write(reinterpret_cast<const char*>(&block_size_), sizeof(block_size_));
  sampling_params->serialize(os);
}

Sequence* Sequence::deserialize(std::istream& is) {
  Config default_config;
  Sequence* seq = new Sequence(default_config.kvcache_block_size, std::vector<int64_t>{});

  size_t task_id_size;
  is.read(reinterpret_cast<char*>(&task_id_size), sizeof(task_id_size));
  seq->task_id.id.resize(task_id_size);
  is.read(reinterpret_cast<char*>(seq->task_id.id.data()), task_id_size);
  is.read(reinterpret_cast<char*>(&seq->last_token), sizeof(seq->last_token));
  is.read(reinterpret_cast<char*>(&seq->num_prompt_tokens_), sizeof(seq->num_prompt_tokens_));
  is.read(reinterpret_cast<char*>(&seq->num_cached_tokens_), sizeof(seq->num_cached_tokens_));

  size_t token_ids_size;
  is.read(reinterpret_cast<char*>(&token_ids_size), sizeof(token_ids_size));
  seq->token_ids_.resize(token_ids_size);
  is.read(reinterpret_cast<char*>(seq->token_ids_.data()), token_ids_size * sizeof(int64_t));

  size_t block_table_size;
  is.read(reinterpret_cast<char*>(&block_table_size), sizeof(block_table_size));
  seq->block_table_.resize(block_table_size);
  is.read(reinterpret_cast<char*>(seq->block_table_.data()), block_table_size * sizeof(int));

  is.read(reinterpret_cast<char*>(&seq->status_), sizeof(seq->status_));
  is.read(reinterpret_cast<char*>(&seq->block_size_), sizeof(seq->block_size_));
  seq->sampling_params.reset(SamplingParams::deserialize(is));
  return seq;
}

}  // namespace llm_engine
