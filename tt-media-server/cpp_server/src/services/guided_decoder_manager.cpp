// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <stdexcept>

#include "utils/concurrent_map.hpp"
#include "utils/logger.hpp"

namespace tt::services {

struct GuidedDecoderManager::Impl {
  xgrammar::TokenizerInfo tokenizerInfo;
  xgrammar::GrammarCompiler compiler;
  int vocabSize;

  struct RequestState {
    xgrammar::GrammarMatcher matcher;
    xgrammar::CompiledGrammar compiledGrammar;
  };

  ConcurrentMap<uint32_t, std::unique_ptr<RequestState>> requests;

  Impl(const std::vector<std::string>& encodedVocab, int vocabSize)
      : tokenizerInfo(encodedVocab, xgrammar::VocabType::BYTE_LEVEL, vocabSize),
        compiler(tokenizerInfo),
        vocabSize(vocabSize) {}
};

GuidedDecoderManager::GuidedDecoderManager(
    const std::vector<std::string>& encodedVocab, int vocabSize)
    : impl(std::make_unique<Impl>(encodedVocab, vocabSize)) {}

GuidedDecoderManager::~GuidedDecoderManager() = default;

void GuidedDecoderManager::initRequest(
    uint32_t taskId,
    const tt::runners::llm_engine::SamplingParams& params) {
  if (!params.hasGuidedDecoding()) return;

  xgrammar::CompiledGrammar compiled = [&]() {
    using tt::config::ResponseFormatType;
    switch (params.response_format_type) {
      case ResponseFormatType::JSON_OBJECT:
        return impl->compiler.CompileBuiltinJSONGrammar();
      case ResponseFormatType::JSON_SCHEMA: {
        if (!params.json_schema_str.has_value()) {
          throw std::invalid_argument(
              "json_schema response format requires a schema string");
        }
        return impl->compiler.CompileJSONSchema(*params.json_schema_str);
      }
      default:
        throw std::logic_error("initRequest called for non-guided request");
    }
  }();

  xgrammar::GrammarMatcher matcher(compiled);

  impl->requests.insert(
      taskId, std::make_unique<Impl::RequestState>(
                  Impl::RequestState{std::move(matcher), std::move(compiled)}));
}

std::vector<int32_t> GuidedDecoderManager::getNextAllowedTokenIds(
    uint32_t taskId) {
  std::vector<int32_t> allowed;
  int vocabSize = impl->vocabSize;

  bool found = impl->requests.modify(
      taskId, [&](std::unique_ptr<Impl::RequestState>& state) {
        int bitmaskSize = xgrammar::GetBitmaskSize(vocabSize);
        std::vector<int32_t> bitmask(bitmaskSize, 0);

        DLTensor tensor;
        tensor.data = bitmask.data();
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
          int wordIdx = i / 32;
          int bitIdx = i % 32;
          if (bitmask[wordIdx] & (1 << bitIdx)) {
            allowed.push_back(i);
          }
        }
      });

  if (!found) return {};
  return allowed;
}

bool GuidedDecoderManager::acceptToken(uint32_t taskId, int32_t tokenId) {
  bool result = true;
  impl->requests.modify(taskId,
                        [&](std::unique_ptr<Impl::RequestState>& state) {
                          result = state->matcher.AcceptToken(tokenId);
                        });
  return result;
}

bool GuidedDecoderManager::isCompleted(uint32_t taskId) const {
  bool completed = false;
  impl->requests.modify(taskId,
                        [&](std::unique_ptr<Impl::RequestState>& state) {
                          completed = state->matcher.IsTerminated();
                        });
  return completed;
}

bool GuidedDecoderManager::hasGuidedDecoding(uint32_t taskId) const {
  return impl->requests.contains(taskId);
}

void GuidedDecoderManager::removeRequest(uint32_t taskId) {
  impl->requests.erase(taskId);
}

}  // namespace tt::services
