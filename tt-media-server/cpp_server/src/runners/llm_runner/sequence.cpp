#include "runners/llm_runner/sequence.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "config/runner_config.hpp"
#include "llm_runner/sampling_params.hpp"

namespace tt::runners::llm_engine {

using Config = tt::config::LLMConfig;

Sequence::Sequence(uint32_t taskId, int blockSize,
                   std::vector<int64_t> inputTokenIds,
                   const SamplingParams& inputSamplingParams)
    : taskId(taskId),
      status(SequenceStatus::WAITING),
      tokenIds(std::move(inputTokenIds)),
      numPromptTokens(tokenIds.size()),
      samplingParams(std::make_unique<SamplingParams>(inputSamplingParams)),
      blockSize(blockSize) {
  if (!tokenIds.empty()) {
    lastToken = tokenIds.back();
  }
}

std::vector<int64_t> Sequence::block(size_t i) const {
  size_t n = numBlocks();
  if (i >= n) {
    throw std::out_of_range("block index out of range");
  }
  size_t start = i * blockSize;
  size_t end = std::min(start + blockSize, tokenIds.size());
  return {tokenIds.begin() + start, tokenIds.begin() + end};
}

std::vector<int64_t> Sequence::completionTokenIds() const {
  if (numPromptTokens >= tokenIds.size()) {
    return {};
  }
  return {tokenIds.begin() + numPromptTokens, tokenIds.end()};
}

void Sequence::appendToken(int64_t tokenId) {
  tokenIds.push_back(tokenId);
  lastToken = tokenId;
}

void Sequence::serialize(std::ostream& os) const {
  size_t tokenIdsSize = tokenIds.size();
  size_t blockTableSize = blockTable.size();
  os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
  os.write(reinterpret_cast<const char*>(&lastToken), sizeof(lastToken));
  os.write(reinterpret_cast<const char*>(&numPromptTokens),
           sizeof(numPromptTokens));
  os.write(reinterpret_cast<const char*>(&numCachedTokens),
           sizeof(numCachedTokens));
  os.write(reinterpret_cast<const char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  os.write(reinterpret_cast<const char*>(tokenIds.data()),
           tokenIdsSize * sizeof(int64_t));
  os.write(reinterpret_cast<const char*>(&blockTableSize),
           sizeof(blockTableSize));
  os.write(reinterpret_cast<const char*>(blockTable.data()),
           blockTableSize * sizeof(int));
  os.write(reinterpret_cast<const char*>(&status), sizeof(status));
  os.write(reinterpret_cast<const char*>(&blockSize), sizeof(blockSize));
  os.write(reinterpret_cast<const char*>(&kvCacheSlot), sizeof(kvCacheSlot));
  os.write(reinterpret_cast<const char*>(&continuation), sizeof(continuation));
  os.write(reinterpret_cast<const char*>(&disaggregated),
           sizeof(disaggregated));
  samplingParams->serialize(os);
}

Sequence Sequence::deserialize(std::istream& is) {
  uint32_t taskId;
  is.read(reinterpret_cast<char*>(&taskId), sizeof(taskId));

  Config defaultConfig;
  Sequence seq(taskId, static_cast<int>(defaultConfig.kvcache_block_size),
               std::vector<int64_t>{});

  is.read(reinterpret_cast<char*>(&seq.lastToken), sizeof(seq.lastToken));
  is.read(reinterpret_cast<char*>(&seq.numPromptTokens),
          sizeof(seq.numPromptTokens));
  is.read(reinterpret_cast<char*>(&seq.numCachedTokens),
          sizeof(seq.numCachedTokens));

  size_t tokenIdsSize;
  is.read(reinterpret_cast<char*>(&tokenIdsSize), sizeof(tokenIdsSize));
  seq.tokenIds.resize(tokenIdsSize);
  is.read(reinterpret_cast<char*>(seq.tokenIds.data()),
          tokenIdsSize * sizeof(int64_t));

  size_t blockTableSize;
  is.read(reinterpret_cast<char*>(&blockTableSize), sizeof(blockTableSize));
  seq.blockTable.resize(blockTableSize);
  is.read(reinterpret_cast<char*>(seq.blockTable.data()),
          blockTableSize * sizeof(int));

  is.read(reinterpret_cast<char*>(&seq.status), sizeof(seq.status));
  is.read(reinterpret_cast<char*>(&seq.blockSize), sizeof(seq.blockSize));
  is.read(reinterpret_cast<char*>(&seq.kvCacheSlot), sizeof(seq.kvCacheSlot));
  is.read(reinterpret_cast<char*>(&seq.continuation), sizeof(seq.continuation));
  is.read(reinterpret_cast<char*>(&seq.disaggregated),
          sizeof(seq.disaggregated));
  seq.samplingParams = SamplingParams::deserialize(is);
  return seq;
}

}  // namespace tt::runners::llm_engine
