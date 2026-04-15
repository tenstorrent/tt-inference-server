// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <optional>
#include <stdexcept>

#include "utils/concurrent_map.hpp"
#include "utils/logger.hpp"

namespace tt::services {

using tt::utils::ConcurrentMap;
using Clock = std::chrono::steady_clock;

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

  ConcurrentMap<uint32_t, std::unique_ptr<RequestState>> requests;

  std::atomic<uint64_t> totalFillBitmaskUs{0};
  std::atomic<uint64_t> totalBitmaskConvertUs{0};
  std::atomic<uint64_t> totalAcceptTokenUs{0};
  std::atomic<uint64_t> totalInitRequestUs{0};
  std::atomic<uint64_t> callCount{0};

  static constexpr uint64_t LOG_INTERVAL = 100;

  Impl(const std::vector<std::string>& encodedVocab, int vocabSize)
      : tokenizerInfo(encodedVocab, xgrammar::VocabType::BYTE_LEVEL, vocabSize),
        compiler(tokenizerInfo),
        vocabSize(vocabSize),
        bitmaskSize(xgrammar::GetBitmaskSize(vocabSize)) {}

  const xgrammar::CompiledGrammar& getJsonObjectGrammar() {
    if (!cachedJsonObjectGrammar.has_value()) {
      cachedJsonObjectGrammar.emplace(compiler.CompileBuiltinJSONGrammar());
    }
    return *cachedJsonObjectGrammar;
  }

  void logProfilingStats() {
    uint64_t count = callCount.load();
    if (count != 1 && (count == 0 || count % LOG_INTERVAL != 0)) return;

    double avgFill = totalFillBitmaskUs.load() / static_cast<double>(count);
    double avgConvert =
        totalBitmaskConvertUs.load() / static_cast<double>(count);
    double avgAccept = totalAcceptTokenUs.load() / static_cast<double>(count);
    uint64_t initUs = totalInitRequestUs.load();

    FILE* f = fopen("/tmp/guided_decoder_profiling.log", "a");
    if (f) {
      fprintf(f,
              "[GuidedDecoder profiling] after %lu calls: "
              "FillBitmask=%.1fus, BitmaskConvert=%.1fus, AcceptToken=%.1fus, "
              "total_per_token=%.1fus (%.2fms), initRequest_total=%luus\n",
              count, avgFill, avgConvert, avgAccept,
              avgFill + avgConvert + avgAccept,
              (avgFill + avgConvert + avgAccept) / 1000.0, initUs);
      fclose(f);
    }
    TT_LOG_INFO(
        "[GuidedDecoder profiling] after {} calls: "
        "FillBitmask={:.1f}us, BitmaskConvert={:.1f}us, AcceptToken={:.1f}us, "
        "total_per_token={:.1f}us ({:.2f}ms), initRequest_total={:.0f}us",
        count, avgFill, avgConvert, avgAccept, avgFill + avgConvert + avgAccept,
        (avgFill + avgConvert + avgAccept) / 1000.0,
        static_cast<double>(initUs));
  }
};

GuidedDecoderManager::GuidedDecoderManager(
    const std::vector<std::string>& encodedVocab, int vocabSize)
    : impl(std::make_unique<Impl>(encodedVocab, vocabSize)) {}

GuidedDecoderManager::~GuidedDecoderManager() = default;

void GuidedDecoderManager::initRequest(
    uint32_t taskId, const tt::runners::llm_engine::SamplingParams& params) {
  if (!params.hasGuidedDecoding()) return;

  auto t0 = Clock::now();

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

  impl->requests.insert(taskId,
                        std::make_unique<Impl::RequestState>(Impl::RequestState{
                            std::move(matcher), std::move(bitmaskBuffer)}));

  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0);
  impl->totalInitRequestUs.fetch_add(elapsed.count());
}

std::vector<int32_t> GuidedDecoderManager::getNextAllowedTokenIds(
    uint32_t taskId) {
  std::vector<int32_t> allowed;
  int vocabSize = impl->vocabSize;
  int bitmaskSize = impl->bitmaskSize;

  impl->requests.modify(taskId, [&](std::unique_ptr<Impl::RequestState>&
                                        state) {
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

    auto t0 = Clock::now();
    state->matcher.FillNextTokenBitmask(&tensor);
    auto t1 = Clock::now();

    allowed.reserve(1024);
    for (int i = 0; i < vocabSize; ++i) {
      if (state->bitmaskBuffer[i / 32] & (1 << (i % 32))) {
        allowed.push_back(i);
      }
    }
    auto t2 = Clock::now();

    impl->totalFillBitmaskUs.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    impl->totalBitmaskConvertUs.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
  });

  return allowed;
}

TokenAcceptResult GuidedDecoderManager::acceptToken(uint32_t taskId,
                                                    int32_t tokenId) {
  TokenAcceptResult result;
  auto t0 = Clock::now();
  impl->requests.modify(
      taskId, [&](std::unique_ptr<Impl::RequestState>& state) {
        result.accepted = state->matcher.AcceptToken(tokenId);
        result.completed = result.accepted && state->matcher.IsTerminated();
      });
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0);
  impl->totalAcceptTokenUs.fetch_add(elapsed.count());
  impl->callCount.fetch_add(1);
  impl->logProfilingStats();
  return result;
}

bool GuidedDecoderManager::hasGuidedDecoding(uint32_t taskId) const {
  return impl->requests.contains(taskId);
}

void GuidedDecoderManager::removeRequest(uint32_t taskId) {
  impl->requests.erase(taskId);
}

}  // namespace tt::services
