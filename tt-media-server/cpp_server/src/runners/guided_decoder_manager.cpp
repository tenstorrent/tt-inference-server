// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace tt::runners {

using SamplingParams = tt::domain::SamplingParams;

namespace {

// Cap on the number of allowed tokens we evaluate when picking a single best
// token in deterministic mode. Free-choice grammar states (e.g. string
// contents) can have hundreds of allowed tokens; checking each via Fork +
// AcceptToken + FillNextTokenBitmask is O(vocab_size) per candidate, so we
// bound the work for mock-runner test latency. The structural tokens that
// drive termination (",", "}", "\"") almost always sit at low byte-level token
// IDs, so iterating from low to high reaches them quickly in practice.
constexpr int K_MAX_DETERMINISTIC_CANDIDATES = 256;

void initBitmaskTensor(DLTensor& tensor, std::vector<int32_t>& bitmask,
                       int64_t& shapeStorage) {
  tensor.data = bitmask.data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = xgrammar::GetBitmaskDLType();
  shapeStorage = static_cast<int64_t>(bitmask.size());
  tensor.shape = &shapeStorage;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
}

bool isBitmaskBitSet(const std::vector<int32_t>& bitmask, int32_t tokenId) {
  if (tokenId < 0) return false;
  size_t word = static_cast<size_t>(tokenId) / 32;
  if (word >= bitmask.size()) return false;
  return ((static_cast<uint32_t>(bitmask[word]) >> (tokenId % 32)) & 1) != 0;
}

int popcountBitmask(const std::vector<int32_t>& bitmask) {
  int total = 0;
  for (int32_t v : bitmask) {
    total += __builtin_popcount(static_cast<uint32_t>(v));
  }
  return total;
}

/**
 * Count tokens that are allowed in `next` but were not allowed in `current`.
 * A candidate that introduces new options has advanced the grammar's NPDA
 * into a different state; a candidate whose follow-on bitmask is a subset of
 * the current one is effectively a no-op (e.g. whitespace before a forced
 * terminal) and would let the mock loop forever.
 */
int countNewlyAllowedBits(const std::vector<int32_t>& current,
                          const std::vector<int32_t>& next) {
  int total = 0;
  size_t n = std::min(current.size(), next.size());
  for (size_t i = 0; i < n; ++i) {
    uint32_t newBits = static_cast<uint32_t>(next[i]) &
                       ~static_cast<uint32_t>(current[i]);
    total += __builtin_popcount(newBits);
  }
  return total;
}

/**
 * Pick a single token from the current matcher state that drives the grammar
 * toward termination.
 *
 * Strategy (in priority order):
 * 1. If the matcher is already completed, return a stop token.
 * 2. If accepting a candidate immediately completes the grammar, return it.
 * 3. Prefer "progressing" candidates - tokens whose follow-on bitmask
 *    introduces options that were NOT allowed in the current state. This
 *    rejects no-op candidates such as whitespace before a forced terminal,
 *    where the next state is just a subset of the current one.
 * 4. Among progressing candidates, pick the one whose follow-on state has
 *    the smallest set of allowed tokens (closest to forced/terminal).
 * 5. Fall back to any acceptable candidate if no progressing one exists
 *    (degenerate grammars or single-allowed-token states).
 *
 * Returns -1 if no candidate is acceptable.
 */
int32_t pickTerminatingToken(xgrammar::GrammarMatcher& matcher,
                             const std::vector<int32_t>& bitmask,
                             int vocabSize) {
  if (matcher.IsCompleted()) {
    for (int stop : matcher.GetStopTokenIds()) {
      if (isBitmaskBitSet(bitmask, stop)) return static_cast<int32_t>(stop);
    }
  }

  int32_t bestProgressingToken = -1;
  int bestProgressingScore = INT_MAX;
  int32_t bestFallbackToken = -1;
  int bestFallbackScore = INT_MAX;
  int candidatesChecked = 0;

  std::vector<int32_t> followBitmask(bitmask.size(), 0);
  DLTensor followTensor{};
  int64_t followShape = 0;
  initBitmaskTensor(followTensor, followBitmask, followShape);

  for (size_t w = 0; w < bitmask.size(); ++w) {
    uint32_t word = static_cast<uint32_t>(bitmask[w]);
    while (word != 0) {
      int bit = __builtin_ctz(word);
      word &= word - 1;
      int32_t candidate = static_cast<int32_t>(w * 32 + bit);
      if (candidate >= vocabSize) continue;

      auto fork = matcher.Fork();
      if (!fork.AcceptToken(candidate)) continue;

      if (fork.IsCompleted()) {
        // Best possible outcome - a single forced stop token will follow.
        return candidate;
      }

      std::fill(followBitmask.begin(), followBitmask.end(), 0);
      fork.FillNextTokenBitmask(&followTensor);

      int score = popcountBitmask(followBitmask);
      bool advances = countNewlyAllowedBits(bitmask, followBitmask) > 0;

      if (advances) {
        if (score < bestProgressingScore) {
          bestProgressingScore = score;
          bestProgressingToken = candidate;
        }
      } else if (bestProgressingToken < 0 && score < bestFallbackScore) {
        bestFallbackScore = score;
        bestFallbackToken = candidate;
      }

      if (++candidatesChecked >= K_MAX_DETERMINISTIC_CANDIDATES) {
        return bestProgressingToken >= 0 ? bestProgressingToken
                                         : bestFallbackToken;
      }
    }
  }
  return bestProgressingToken >= 0 ? bestProgressingToken : bestFallbackToken;
}

void reduceBitmaskToSingleToken(std::vector<int32_t>& bitmask, int32_t tokenId) {
  std::fill(bitmask.begin(), bitmask.end(), 0);
  bitmask[tokenId / 32] |= 1 << (tokenId % 32);
}

}  // namespace

struct GuidedDecoderManager::Impl {
  xgrammar::TokenizerInfo tokenizerInfo;
  xgrammar::GrammarCompiler compiler;
  int vocabSize;
  int bitmaskSize;
  bool deterministicSelect;

  struct RequestState {
    xgrammar::GrammarMatcher matcher;
  };

  std::unordered_map<uint32_t, std::unique_ptr<RequestState>> requests;

  Impl(const std::vector<std::string>& encodedVocab, int vocabSize,
       bool deterministicSelect)
      : tokenizerInfo(encodedVocab, xgrammar::VocabType::BYTE_LEVEL, vocabSize),
        compiler(tokenizerInfo),
        vocabSize(vocabSize),
        bitmaskSize(xgrammar::GetBitmaskSize(vocabSize)),
        deterministicSelect(deterministicSelect) {}
};

GuidedDecoderManager::GuidedDecoderManager(
    const std::vector<std::string>& encodedVocab, int vocabSize,
    bool deterministicSelect)
    : impl(std::make_unique<Impl>(encodedVocab, vocabSize, deterministicSelect)) {}

GuidedDecoderManager::~GuidedDecoderManager() = default;

void GuidedDecoderManager::initRequest(uint32_t taskId,
                                       const SamplingParams& params) {
  if (!params.hasGuidedDecoding()) return;

  using tt::config::ResponseFormatType;
  xgrammar::CompiledGrammar compiled = [&] {
    switch (params.response_format_type) {
      case ResponseFormatType::JSON_OBJECT:
        return impl->compiler.CompileBuiltinJSONGrammar();
      case ResponseFormatType::JSON_SCHEMA:
        if (!params.json_schema_str.has_value()) {
          throw std::invalid_argument(
              "json_schema response format requires a schema string");
        }
        return impl->compiler.CompileJSONSchema(*params.json_schema_str);
      default:
        throw std::logic_error("initRequest called for non-guided request");
    }
  }();

  impl->requests[taskId] = std::make_unique<Impl::RequestState>(
      Impl::RequestState{xgrammar::GrammarMatcher(compiled)});
}

void GuidedDecoderManager::fillNextBitmask(uint32_t taskId,
                                           std::vector<int32_t>& bitmask) {
  auto it = impl->requests.find(taskId);
  if (it == impl->requests.end()) return;

  bitmask.assign(impl->bitmaskSize, 0);

  DLTensor tensor{};
  int64_t shape = 0;
  initBitmaskTensor(tensor, bitmask, shape);

  auto& matcher = it->second->matcher;
  matcher.FillNextTokenBitmask(&tensor);

  if (impl->deterministicSelect) {
    int32_t chosen = pickTerminatingToken(matcher, bitmask, impl->vocabSize);
    if (chosen >= 0) {
      reduceBitmaskToSingleToken(bitmask, chosen);
    }
  }
}

int GuidedDecoderManager::vocabSize() const { return impl->vocabSize; }

int GuidedDecoderManager::bitmaskSize() const { return impl->bitmaskSize; }

TokenAcceptResult GuidedDecoderManager::acceptToken(uint32_t taskId,
                                                    int32_t tokenId) {
  TokenAcceptResult result;
  auto it = impl->requests.find(taskId);
  if (it != impl->requests.end()) {
    result.accepted = it->second->matcher.AcceptToken(tokenId);
    result.completed = result.accepted && it->second->matcher.IsTerminated();
  }
  return result;
}

bool GuidedDecoderManager::hasGuidedDecoding(uint32_t taskId) const {
  return impl->requests.count(taskId) > 0;
}

void GuidedDecoderManager::removeRequest(uint32_t taskId) {
  impl->requests.erase(taskId);
}

}  // namespace tt::runners
