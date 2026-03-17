#include "runners/llm_runner/sequence.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "config/runner_config.hpp"
#include "llm_runner/sampling_params.hpp"

namespace llm_engine {

using Config = tt::config::LLMConfig;

Sequence::Sequence(TaskID taskId, int blockSize, std::vector<int64_t> tokenIds,
                   const SamplingParams& samplingParams)
    : task_id(std::move(taskId)),
      status_(SequenceStatus::WAITING),
      token_ids_(std::move(tokenIds)),
      num_prompt_tokens_(token_ids_.size()),
      sampling_params(std::make_unique<SamplingParams>(samplingParams)),
      block_size_(blockSize) {
  assert(!this->task_id.id.empty() && "Sequence requires a non-empty task_id");
  if (!token_ids_.empty()) {
    last_token = token_ids_.back();
  }
}

std::vector<int64_t> Sequence::block(size_t i) const {
  size_t n = numBlocks();
  if (i >= n) {
    throw std::out_of_range("block index out of range");
  }
  size_t start = i * block_size_;
  size_t end = std::min(start + block_size_, token_ids_.size());
  return std::vector<int64_t>(token_ids_.begin() + start,
                              token_ids_.begin() + end);
}

std::vector<int64_t> Sequence::completionTokenIds() const {
  if (num_prompt_tokens_ >= token_ids_.size()) {
    return {};
  }
  return std::vector<int64_t>(token_ids_.begin() + num_prompt_tokens_,
                              token_ids_.end());
}

void Sequence::appendToken(int64_t tokenId) {
  token_ids_.push_back(tokenId);
  last_token = tokenId;
}

void Sequence::serialize(std::ostream& os) const {
  size_t tokenIdsSize = token_ids_.size();
  size_t blockTableSize = block_table_.size();
  size_t idSize = task_id.id.size();
  os.write(reinterpret_cast<const char*>(&idSize), sizeof(idSize));
  os.write(task_id.id.c_str(), idSize);
  os.write(reinterpret_cast<const char*>(&last_token), sizeof(last_token));
  os.write(reinterpret_cast<const char*>(&num_prompt_tokens_),
           sizeof(num_prompt_tokens_));
  os.write(reinterpret_cast<const char*>(&num_cached_tokens_),
           sizeof(num_cached_tokens_));
  os.write(reinterpret_cast<const char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  os.write(reinterpret_cast<const char*>(token_ids_.data()),
           tokenIdsSize * sizeof(int64_t));
  os.write(reinterpret_cast<const char*>(&blockTableSize),
           sizeof(blockTableSize));
  os.write(reinterpret_cast<const char*>(block_table_.data()),
           blockTableSize * sizeof(int));
  os.write(reinterpret_cast<const char*>(&status_), sizeof(status_));
  os.write(reinterpret_cast<const char*>(&block_size_), sizeof(block_size_));
  sampling_params->serialize(os);
}

Sequence* Sequence::deserialize(std::istream& is) {
  size_t taskIdSize;
  is.read(reinterpret_cast<char*>(&taskIdSize), sizeof(taskIdSize));
  std::string taskIdStr(taskIdSize, '\0');
  is.read(taskIdStr.data(), taskIdSize);

  Config defaultConfig;
  Sequence* seq =
      new Sequence(TaskID(std::move(taskIdStr)),
                   defaultConfig.kvcache_block_size, std::vector<int64_t>{});

  is.read(reinterpret_cast<char*>(&seq->last_token), sizeof(seq->last_token));
  is.read(reinterpret_cast<char*>(&seq->num_prompt_tokens_),
          sizeof(seq->num_prompt_tokens_));
  is.read(reinterpret_cast<char*>(&seq->num_cached_tokens_),
          sizeof(seq->num_cached_tokens_));

  size_t tokenIdsSize;
  is.read(reinterpret_cast<char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  seq->token_ids_.resize(tokenIdsSize);
  is.read(reinterpret_cast<char*>(seq->token_ids_.data()),
          tokenIdsSize * sizeof(int64_t));

  size_t blockTableSize;
  is.read(reinterpret_cast<char*>(&blockTableSize), sizeof(blockTableSize));
  seq->block_table_.resize(blockTableSize);
  is.read(reinterpret_cast<char*>(seq->block_table_.data()),
          blockTableSize * sizeof(int));

  is.read(reinterpret_cast<char*>(&seq->status_), sizeof(seq->status_));
  is.read(reinterpret_cast<char*>(&seq->block_size_), sizeof(seq->block_size_));
  seq->sampling_params.reset(SamplingParams::deserialize(is));
  return seq;
}

}  // namespace llm_engine
