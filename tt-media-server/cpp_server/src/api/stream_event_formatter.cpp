// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/stream_event_formatter.hpp"

#include "domain/llm/chat_completion_response.hpp"

namespace tt::api {

// -- ChatCompletionEventFormatter --------------------------------------------

std::string ChatCompletionEventFormatter::formatInitialEvents(
    const ResponseWriterParams& params,
    const std::optional<CompletionUsage>& initialUsage) {
  auto initialChunk = ChatCompletionStreamChunk::makeInitialChunk(
      params.completionId, params.model, params.created, initialUsage);
  return initialChunk.toSSE();
}

std::string ChatCompletionEventFormatter::formatTokenEvents(
    const ResponseWriterParams& params, const LLMStreamChunk& chunk,
    const std::optional<CompletionUsage>& usage, int /*currentTokens*/,
    const std::string& /*accumulatedText*/) {
  auto streamChunk = ChatCompletionStreamChunk::makeContentChunk(
      params.completionId, params.model, params.created, chunk.choices[0],
      usage);
  return streamChunk.toSSE();
}

std::string ChatCompletionEventFormatter::formatFinalEvents(
    const ResponseWriterParams& params, const CompletionUsage& usage,
    const std::string& /*accumulatedText*/,
    const std::optional<std::string>& /*finishReason*/, bool includeUsage) {
  std::string out;
  if (includeUsage) {
    out += ChatCompletionStreamChunk::makeUsageChunk(
               params.completionId, params.model, params.created, usage)
               .toSSE();
  }
  out += "data: [DONE]\n\n";
  return out;
}

}  // namespace tt::api
