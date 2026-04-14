// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <optional>
#include <stdexcept>

#include "utils/concurrent_map.hpp"

namespace tt::services {

using tt::utils::ConcurrentMap;

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

  impl->requests.insert(taskId,
                        std::make_unique<Impl::RequestState>(Impl::RequestState{
                            std::move(matcher), std::move(bitmaskBuffer)}));
}

std::vector<int32_t> GuidedDecoderManager::getNextAllowedTokenIds(
    uint32_t taskId) {
  std::vector<int32_t> allowed;
  int vocabSize = impl->vocabSize;
  int bitmaskSize = impl->bitmaskSize;

  impl->requests.modify(
      taskId, [&](std::unique_ptr<Impl::RequestState>& state) {
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

        allowed.reserve(1024);
        for (int i = 0; i < vocabSize; ++i) {
          if (state->bitmaskBuffer[i / 32] & (1 << (i % 32))) {
            allowed.push_back(i);
          }
        }
      });

  return allowed;
}

TokenAcceptResult GuidedDecoderManager::acceptToken(uint32_t taskId,
                                                    int32_t tokenId) {
  TokenAcceptResult result;
  impl->requests.modify(
      taskId, [&](std::unique_ptr<Impl::RequestState>& state) {
        result.accepted = state->matcher.AcceptToken(tokenId);
        result.completed = result.accepted && state->matcher.IsTerminated();
      });
  return result;
}

bool GuidedDecoderManager::hasGuidedDecoding(uint32_t taskId) const {
  return impl->requests.contains(taskId);
}

void GuidedDecoderManager::removeRequest(uint32_t taskId) {
  impl->requests.erase(taskId);
}

}  // namespace tt::services
