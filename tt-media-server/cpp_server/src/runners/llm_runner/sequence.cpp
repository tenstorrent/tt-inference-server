#include "runners/llm_runner/sequence.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "config/runner_config.hpp"
#include "llm_runner/sampling_params.hpp"

namespace llm_engine {

using Config = tt::config::LLMConfig;

Sequence::Sequence(uint32_t taskId, int blockSize,
                   std::vector<int64_t> tokenIds,
                   const SamplingParams& samplingParams)
    : taskId(taskId),
      status_(SequenceStatus::WAITING),
      tokenIds_(std::move(tokenIds)),
      numPromptTokens_(tokenIds_.size()),
      samplingParams_(std::make_unique<SamplingParams>(samplingParams)),
      blockSize_(blockSize) {
  if (!tokenIds_.empty()) {
    lastToken_ = tokenIds_.back();
  }
}

std::vector<int64_t> Sequence::block(size_t i) const {
  size_t n = numBlocks();
  if (i >= n) {
    throw std::out_of_range("block index out of range");
  }
  size_t start = i * blockSize_;
  size_t end = std::min(start + blockSize_, tokenIds_.size());
  return {tokenIds_.begin() + start, tokenIds_.begin() + end};
}

std::vector<int64_t> Sequence::completionTokenIds() const {
  if (numPromptTokens_ >= tokenIds_.size()) {
    return {};
  }
  return {tokenIds_.begin() + numPromptTokens_, tokenIds_.end()};
}

void Sequence::appendToken(int64_t tokenId) {
  tokenIds_.push_back(tokenId);
  lastToken_ = tokenId;
}

void Sequence::serialize(std::ostream& os) const {
  size_t tokenIdsSize = tokenIds_.size();
  size_t blockTableSize = blockTable_.size();
  os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
  os.write(reinterpret_cast<const char*>(&lastToken_), sizeof(lastToken_));
  os.write(reinterpret_cast<const char*>(&numPromptTokens_),
           sizeof(numPromptTokens_));
  os.write(reinterpret_cast<const char*>(&numCachedTokens_),
           sizeof(numCachedTokens_));
  os.write(reinterpret_cast<const char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  os.write(reinterpret_cast<const char*>(tokenIds_.data()),
           tokenIdsSize * sizeof(int64_t));
  os.write(reinterpret_cast<const char*>(&blockTableSize),
           sizeof(blockTableSize));
  os.write(reinterpret_cast<const char*>(blockTable_.data()),
           blockTableSize * sizeof(int));
  os.write(reinterpret_cast<const char*>(&status_), sizeof(status_));
  os.write(reinterpret_cast<const char*>(&blockSize_), sizeof(blockSize_));
  os.write(reinterpret_cast<const char*>(&address_), sizeof(address_));
  samplingParams_->serialize(os);
}

Sequence Sequence::deserialize(std::istream& is) {
  uint32_t taskId;
  is.read(reinterpret_cast<char*>(&taskId), sizeof(taskId));

  Config defaultConfig;
  Sequence seq(taskId, static_cast<int>(defaultConfig.kvcache_block_size),
               std::vector<int64_t>{});

  is.read(reinterpret_cast<char*>(&seq.lastToken_), sizeof(seq.lastToken_));
  is.read(reinterpret_cast<char*>(&seq.numPromptTokens_),
          sizeof(seq.numPromptTokens_));
  is.read(reinterpret_cast<char*>(&seq.numCachedTokens_),
          sizeof(seq.numCachedTokens_));

  size_t tokenIdsSize;
  is.read(reinterpret_cast<char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  seq.tokenIds_.resize(tokenIdsSize);
  is.read(reinterpret_cast<char*>(seq.tokenIds_.data()),
          tokenIdsSize * sizeof(int64_t));

  size_t blockTableSize;
  is.read(reinterpret_cast<char*>(&blockTableSize), sizeof(blockTableSize));
  seq.blockTable_.resize(blockTableSize);
  is.read(reinterpret_cast<char*>(seq.blockTable_.data()),
          blockTableSize * sizeof(int));

  is.read(reinterpret_cast<char*>(&seq.status_), sizeof(seq.status_));
  is.read(reinterpret_cast<char*>(&seq.blockSize_), sizeof(seq.blockSize_));
  is.read(reinterpret_cast<char*>(&seq.address_), sizeof(seq.address_));
  seq.samplingParams_ = SamplingParams::deserialize(is);
  return seq;
}

}  // namespace llm_engine
