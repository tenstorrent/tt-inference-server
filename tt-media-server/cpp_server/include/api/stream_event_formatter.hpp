// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "api/response_writer/response_writer.hpp"
#include "domain/llm/llm_response.hpp"

namespace tt::api {

using namespace tt::domain::llm;

/**
 * Strategy interface that converts LLM stream callbacks into SSE frames for a
 * specific OpenAI-compatible protocol (chat completions, ...).
 *
 * The writer owns one formatter per request and calls these hooks in this
 * order: formatInitialEvents -> formatTokenEvents (N times) ->
 * formatFinalEvents.
 */
class StreamEventFormatter {
 public:
  virtual ~StreamEventFormatter() = default;

  /** SSE emitted once, right before the first content token. */
  virtual std::string formatInitialEvents(
      const ResponseWriterParams& params,
      const std::optional<CompletionUsage>& initialUsage) = 0;

  /** SSE emitted for a single streaming chunk from the LLM service. */
  virtual std::string formatTokenEvents(
      const ResponseWriterParams& params, const LLMStreamChunk& chunk,
      const std::optional<CompletionUsage>& usage, int currentTokens,
      const std::string& accumulatedText) = 0;

  /** SSE emitted at finalization (usage chunk + done marker / completed). */
  virtual std::string formatFinalEvents(
      const ResponseWriterParams& params, const CompletionUsage& usage,
      const std::string& accumulatedText,
      const std::optional<std::string>& finishReason, bool includeUsage) = 0;
};

/** Chat completions SSE (current default): emits `chat.completion.chunk`
 *  objects and terminates with `data: [DONE]`. */
class ChatCompletionEventFormatter final : public StreamEventFormatter {
 public:
  std::string formatInitialEvents(
      const ResponseWriterParams& params,
      const std::optional<CompletionUsage>& initialUsage) override;

  std::string formatTokenEvents(const ResponseWriterParams& params,
                                const LLMStreamChunk& chunk,
                                const std::optional<CompletionUsage>& usage,
                                int currentTokens,
                                const std::string& accumulatedText) override;

  std::string formatFinalEvents(const ResponseWriterParams& params,
                                const CompletionUsage& usage,
                                const std::string& accumulatedText,
                                const std::optional<std::string>& finishReason,
                                bool includeUsage) override;
};

}  // namespace tt::api
