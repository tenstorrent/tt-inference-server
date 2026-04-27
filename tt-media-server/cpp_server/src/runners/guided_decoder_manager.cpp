// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/guided_decoder_manager.hpp"

#include <xgrammar/xgrammar.h>

#include <stdexcept>
#include <unordered_map>

namespace tt::runners {

using SamplingParams = tt::domain::SamplingParams;

struct GuidedDecoderManager::Impl {
  xgrammar::TokenizerInfo tokenizerInfo;
  xgrammar::GrammarCompiler compiler;
  int vocabSize;
  int bitmaskSize;

  struct RequestState {
    xgrammar::GrammarMatcher matcher;
  };

  std::unordered_map<uint32_t, std::unique_ptr<RequestState>> requests;

  Impl(const std::vector<std::string>& encodedVocab, int vocabSize)
      : tokenizerInfo(encodedVocab, xgrammar::VocabType::BYTE_LEVEL, vocabSize),
        compiler(tokenizerInfo),
        vocabSize(vocabSize),
        bitmaskSize(xgrammar::GetBitmaskSize(vocabSize)) {}
};

GuidedDecoderManager::GuidedDecoderManager(
    const std::vector<std::string>& encodedVocab, int vocabSize)
    : impl(std::make_unique<Impl>(encodedVocab, vocabSize)) {}

GuidedDecoderManager::~GuidedDecoderManager() = default;

void GuidedDecoderManager::initRequest(uint32_t taskId,
                                       const SamplingParams& params) {
  using tt::config::ResponseFormatType;
  ResponseFormatType formatType = params.response_format_type;
  std::optional<std::string> schemaStr = params.json_schema_str;

  if (params.tool_choice.has_value() &&
      params.tool_choice->type == "function" &&
      params.tool_choice->function.has_value() && params.tools.has_value()) {
    const auto& functionName = params.tool_choice->function.value();
    for (const auto& tool : params.tools.value()) {
      if (tool.functionDefinition.name == functionName) {
        formatType = ResponseFormatType::JSON_SCHEMA;

        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        schemaStr =
            Json::writeString(writer, tool.functionDefinition.parameters);
        break;
      }
    }
  }

  if (formatType == ResponseFormatType::TEXT) return;

  xgrammar::CompiledGrammar compiled = [&] {
    switch (formatType) {
      case ResponseFormatType::JSON_OBJECT:
        return impl->compiler.CompileBuiltinJSONGrammar();
      case ResponseFormatType::JSON_SCHEMA:
        if (!schemaStr.has_value()) {
          throw std::invalid_argument(
              "json_schema response format requires a schema string");
        }
        return impl->compiler.CompileJSONSchema(*schemaStr);
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
  tensor.data = bitmask.data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = xgrammar::GetBitmaskDLType();
  int64_t shape = impl->bitmaskSize;
  tensor.shape = &shape;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;

  it->second->matcher.FillNextTokenBitmask(&tensor);
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
