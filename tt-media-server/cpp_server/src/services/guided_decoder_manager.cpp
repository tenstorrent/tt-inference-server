// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "utils/thread_pool.hpp"

namespace tt::services {

struct GuidedDecoderManager::Impl {
  xgrammar::TokenizerInfo tokenizerInfo;
  xgrammar::GrammarCompiler compiler;
  xgrammar::BatchGrammarMatcher batchMatcher;
  int vocabSize;
  int bitmaskSize;

  struct RequestState {
    xgrammar::GrammarMatcher matcher;
  };

  // All access is from the single worker-process inference thread.
  std::unordered_map<uint32_t, std::unique_ptr<RequestState>> requests;
  tt::utils::ThreadPool pool;

  Impl(const std::vector<std::string>& encodedVocab, int vocabSize)
      : tokenizerInfo(encodedVocab, xgrammar::VocabType::BYTE_LEVEL, vocabSize),
        compiler(tokenizerInfo),
        vocabSize(vocabSize),
        bitmaskSize(xgrammar::GetBitmaskSize(vocabSize)),
        pool(std::thread::hardware_concurrency()) {}

  static void extractAllowedIds(const int32_t* bitmaskRow,
                                std::vector<int32_t>& out, int vocabSize,
                                int bitmaskSize) {
    out.reserve(1024);
    for (int w = 0; w < bitmaskSize; ++w) {
      auto word = static_cast<uint32_t>(bitmaskRow[w]);
      while (word != 0) {
        int bit = __builtin_ctz(word);
        int tokenId = w * 32 + bit;
        if (tokenId < vocabSize) out.push_back(tokenId);
        word &= word - 1;
      }
    }
  }
};

GuidedDecoderManager::GuidedDecoderManager(
    const std::vector<std::string>& encodedVocab, int vocabSize)
    : impl(std::make_unique<Impl>(encodedVocab, vocabSize)) {}

GuidedDecoderManager::~GuidedDecoderManager() = default;

void GuidedDecoderManager::initRequest(
    uint32_t taskId, const tt::runners::llm_engine::SamplingParams& params) {
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

std::vector<int32_t> GuidedDecoderManager::getNextAllowedTokenIds(
    uint32_t taskId) {
  auto it = impl->requests.find(taskId);
  if (it == impl->requests.end()) return {};

  std::vector<int32_t> buffer(impl->bitmaskSize, 0);
  DLTensor tensor{};
  tensor.data = buffer.data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = xgrammar::GetBitmaskDLType();
  int64_t shape = impl->bitmaskSize;
  tensor.shape = &shape;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;

  it->second->matcher.FillNextTokenBitmask(&tensor);

  std::vector<int32_t> allowed;
  Impl::extractAllowedIds(buffer.data(), allowed, impl->vocabSize,
                          impl->bitmaskSize);
  return allowed;
}

std::vector<BatchAllowedResult>
GuidedDecoderManager::getNextAllowedTokenIdsBatch(
    const std::vector<uint32_t>& taskIds) {
  int vocabSize = impl->vocabSize;
  int bitmaskSize = impl->bitmaskSize;

  struct WorkItem {
    uint32_t taskId;
    Impl::RequestState* state;
  };
  std::vector<WorkItem> items;
  std::vector<xgrammar::GrammarMatcher> matchers;
  items.reserve(taskIds.size());
  matchers.reserve(taskIds.size());

  for (uint32_t id : taskIds) {
    auto it = impl->requests.find(id);
    if (it != impl->requests.end()) {
      items.push_back({id, it->second.get()});
      matchers.push_back(std::move(it->second->matcher));
    }
  }

  if (items.empty()) return {};

  auto batchSize = static_cast<int64_t>(matchers.size());

  // Batch-fill bitmasks using xgrammar's internal thread pool
  std::vector<int32_t> batchBuffer(batchSize * bitmaskSize, 0);
  int64_t shape[2] = {batchSize, static_cast<int64_t>(bitmaskSize)};
  DLTensor batchTensor{};
  batchTensor.data = batchBuffer.data();
  batchTensor.device = {kDLCPU, 0};
  batchTensor.ndim = 2;
  batchTensor.dtype = xgrammar::GetBitmaskDLType();
  batchTensor.shape = shape;
  batchTensor.strides = nullptr;
  batchTensor.byte_offset = 0;

  impl->batchMatcher.BatchFillNextTokenBitmask(&matchers, &batchTensor);

  // Move matchers back to request states
  for (size_t i = 0; i < items.size(); ++i) {
    items[i].state->matcher = std::move(matchers[i]);
  }

  // Extract allowed token IDs in parallel using thread pool
  std::vector<BatchAllowedResult> results(items.size());

  constexpr size_t kParallelThreshold = 4;
  if (items.size() <= kParallelThreshold) {
    for (size_t i = 0; i < items.size(); ++i) {
      results[i].taskId = items[i].taskId;
      Impl::extractAllowedIds(batchBuffer.data() + i * bitmaskSize,
                              results[i].allowedTokenIds, vocabSize,
                              bitmaskSize);
    }
    return results;
  }

  std::atomic<size_t> completed{0};
  std::mutex doneMutex;
  std::condition_variable doneCv;

  for (size_t i = 0; i < items.size(); ++i) {
    impl->pool.submit([&, i] {
      results[i].taskId = items[i].taskId;
      Impl::extractAllowedIds(batchBuffer.data() + i * bitmaskSize,
                              results[i].allowedTokenIds, vocabSize,
                              bitmaskSize);
      if (completed.fetch_add(1, std::memory_order_acq_rel) + 1 ==
          items.size()) {
        std::lock_guard lk(doneMutex);
        doneCv.notify_one();
      }
    });
  }

  std::unique_lock lk(doneMutex);
  doneCv.wait(lk, [&] {
    return completed.load(std::memory_order_acquire) == items.size();
  });

  return results;
}

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

}  // namespace tt::services
