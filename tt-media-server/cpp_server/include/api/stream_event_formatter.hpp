// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "api/response_writer/response_writer.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/llm/sampling_params.hpp"
#include "domain/responses_request.hpp"

namespace tt::api {

using namespace tt::domain::llm;

/**
 * Strategy interface that converts LLM stream callbacks into SSE frames for a
 * specific OpenAI-compatible protocol (chat completions, responses, ...).
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

/** Responses API SSE: emits `response.created`, `response.output_text.delta`,
 *  `response.completed` / `response.incomplete`, etc. No `[DONE]` terminator.
 */
class ResponsesEventFormatter final : public StreamEventFormatter {
 public:
  ResponsesEventFormatter(std::shared_ptr<domain::ResponsesRequest> request,
                          tt::domain::llm::SamplingParams samplingParams);

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

 private:
  std::shared_ptr<domain::ResponsesRequest> request_;
  tt::domain::llm::SamplingParams sampling_params_;

  static constexpr int kOutputIndex = 0;
  static constexpr int kContentIndex = 0;
  static constexpr const char* kItemId = "msg_0";

  int sequence_number_ = 0;

  std::string formatEvent(const std::string& eventName,
                          const Json::Value& payload);
  std::string buildResponseObjectJson(
      int64_t createdAt, const std::string& status, const Json::Value& output,
      const std::optional<CompletionUsage>& usage) const;
};

}  // namespace tt::api
