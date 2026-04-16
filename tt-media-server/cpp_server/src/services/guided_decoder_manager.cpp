// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <atomic>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "utils/thread_pool.hpp"

namespace tt::services {

struct GuidedDecoderManager::Impl {
  xgrammar::TokenizerInfo tokenizerInfo;
  xgrammar::GrammarCompiler compiler;
  int vocabSize;
  int bitmaskSize;

  std::optional<xgrammar::CompiledGrammar> cachedJsonObjectGrammar;

  struct RequestState {
    xgrammar::GrammarMatcher matcher;
    std::vector<int32_t> bitmaskBuffer;
  };

  std::unordered_map<uint32_t, std::unique_ptr<RequestState>> requests;
  tt::utils::ThreadPool pool;

  Impl(const std::vector<std::string>& encodedVocab, int vocabSize)
      : tokenizerInfo(encodedVocab, xgrammar::VocabType::BYTE_LEVEL, vocabSize),
        compiler(tokenizerInfo),
        vocabSize(vocabSize),
        bitmaskSize(xgrammar::GetBitmaskSize(vocabSize)),
        pool(std::thread::hardware_concurrency()) {}

  const xgrammar::CompiledGrammar& getJsonObjectGrammar() {
    if (!cachedJsonObjectGrammar.has_value()) {
      cachedJsonObjectGrammar.emplace(compiler.CompileBuiltinJSONGrammar());
    }
    return *cachedJsonObjectGrammar;
  }

  static void fillAllowed(RequestState* state, std::vector<int32_t>& out,
                          int vocabSize, int bitmaskSize) {
    std::fill(state->bitmaskBuffer.begin(), state->bitmaskBuffer.end(), 0);

    DLTensor tensor;
    tensor.data = state->bitmaskBuffer.data();
    tensor.device = {kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = xgrammar::GetBitmaskDLType();
    int64_t shape = bitmaskSize;
    tensor.shape = &shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;

    state->matcher.FillNextTokenBitmask(&tensor);

    out.reserve(1024);
    for (int w = 0; w < bitmaskSize; ++w) {
      auto word = static_cast<uint32_t>(state->bitmaskBuffer[w]);
      while (word != 0) {
        int bit = __builtin_ctz(word);
        int tokenId = w * 32 + bit;
        if (tokenId < vocabSize) {
          out.push_back(tokenId);
        }
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
  const xgrammar::CompiledGrammar& compiled =
      [&]() -> const xgrammar::CompiledGrammar& {
    switch (params.response_format_type) {
      case ResponseFormatType::JSON_OBJECT:
        return impl->getJsonObjectGrammar();
      case ResponseFormatType::JSON_SCHEMA: {
        if (!params.json_schema_str.has_value()) {
          throw std::invalid_argument(
              "json_schema response format requires a schema string");
        }
        // json_schema grammars are compiled per request (schema varies)
        // TODO: cache by schema string hash for repeated schemas
        static thread_local xgrammar::CompiledGrammar schemaGrammar =
            impl->compiler.CompileJSONSchema(*params.json_schema_str);
        schemaGrammar =
            impl->compiler.CompileJSONSchema(*params.json_schema_str);
        return schemaGrammar;
      }
      default:
        throw std::logic_error("initRequest called for non-guided request");
    }
  }();

  xgrammar::GrammarMatcher matcher(compiled);
  std::vector<int32_t> bitmaskBuffer(impl->bitmaskSize, 0);

  impl->requests[taskId] = std::make_unique<Impl::RequestState>(
      Impl::RequestState{std::move(matcher), std::move(bitmaskBuffer)});
}

std::vector<int32_t> GuidedDecoderManager::getNextAllowedTokenIds(
    uint32_t taskId) {
  std::vector<int32_t> allowed;
  auto it = impl->requests.find(taskId);
  if (it == impl->requests.end()) return allowed;
  Impl::fillAllowed(it->second.get(), allowed, impl->vocabSize,
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
  items.reserve(taskIds.size());
  for (uint32_t id : taskIds) {
    auto it = impl->requests.find(id);
    if (it != impl->requests.end()) {
      items.push_back({id, it->second.get()});
    }
  }

  std::vector<BatchAllowedResult> results(items.size());

  constexpr size_t kParallelThreshold = 4;
  if (items.size() <= kParallelThreshold) {
    for (size_t i = 0; i < items.size(); ++i) {
      results[i].taskId = items[i].taskId;
      Impl::fillAllowed(items[i].state, results[i].allowedTokenIds, vocabSize,
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
      Impl::fillAllowed(items[i].state, results[i].allowedTokenIds, vocabSize,
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
