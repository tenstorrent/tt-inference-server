// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <mutex>
#include <stdexcept>

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

  mutable std::mutex mu;
  std::unordered_map<uint32_t, RequestState> requests;

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
    uint32_t taskId, const llm_engine::SamplingParams& params) {
  if (!params.hasGuidedDecoding()) return;

  xgrammar::CompiledGrammar compiled = [&]() {
    switch (params.response_format_type) {
      case llm_engine::ResponseFormatType::JSON_OBJECT:
        return impl->compiler.CompileBuiltinJSONGrammar();
      case llm_engine::ResponseFormatType::JSON_SCHEMA: {
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

  std::lock_guard<std::mutex> lock(impl->mu);
  impl->requests.emplace(
      taskId, Impl::RequestState{std::move(matcher), std::move(compiled)});
}

std::vector<int32_t> GuidedDecoderManager::getNextAllowedTokenIds(
    uint32_t taskId) {
  std::lock_guard<std::mutex> lock(impl->mu);
  auto it = impl->requests.find(taskId);
  if (it == impl->requests.end()) return {};

  auto& matcher = it->second.matcher;
  int bitmaskSize = xgrammar::GetBitmaskSize(impl->vocabSize);

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

  matcher.FillNextTokenBitmask(&tensor);

  std::vector<int32_t> allowed;
  allowed.reserve(1024);
  for (int i = 0; i < impl->vocabSize; ++i) {
    int wordIdx = i / 32;
    int bitIdx = i % 32;
    if (bitmask[wordIdx] & (1 << bitIdx)) {
      allowed.push_back(i);
    }
  }

  return allowed;
}

bool GuidedDecoderManager::acceptToken(uint32_t taskId, int32_t tokenId) {
  std::lock_guard<std::mutex> lock(impl->mu);
  auto it = impl->requests.find(taskId);
  if (it == impl->requests.end()) return true;
  return it->second.matcher.AcceptToken(tokenId);
}

bool GuidedDecoderManager::isCompleted(uint32_t taskId) const {
  std::lock_guard<std::mutex> lock(impl->mu);
  auto it = impl->requests.find(taskId);
  if (it == impl->requests.end()) return false;
  return it->second.matcher.IsTerminated();
}

bool GuidedDecoderManager::hasGuidedDecoding(uint32_t taskId) const {
  std::lock_guard<std::mutex> lock(impl->mu);
  return impl->requests.count(taskId) > 0;
}

void GuidedDecoderManager::removeRequest(uint32_t taskId) {
  std::lock_guard<std::mutex> lock(impl->mu);
  impl->requests.erase(taskId);
}

}  // namespace tt::services
