// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cmath>
#include <optional>
#include <sstream>
#include <string>

#include "api/response_writer/response_writer.hpp"
#include "domain/llm/llm_response.hpp"

namespace tt::api {

/**
 * Pure data container that buffers LLMStreamChunks for a non-streaming chat
 * completion request and assembles them into one LLMResponse. The controller
 * owns an instance per non-streaming request and feeds it from the streaming
 * sink callback; on the final chunk the controller calls build(), runs
 * service-level postProcess, and sends a single HTTP JSON response.
 *
 * Holds no service, HTTP, or session references -- it only knows how to
 * append chunk content and emit an aggregated response. Thread-safety is
 * provided by the LLMService consumer thread serializing sink invocations,
 * not by this struct.
 */
struct LLMResponseAccumulator {
  void add(const tt::domain::llm::LLMStreamChunk& chunk) {
    if (chunk.choices.empty()) return;

    const auto& choice = chunk.choices[0];
    if (choice.reasoning.has_value()) {
      reasoning << choice.reasoning.value();
      hasReasoning = true;
    }
    answer << choice.text;

    if (!choice.text.empty() || choice.reasoning.has_value()) {
      noteToken();
    }

    if (choice.finish_reason.has_value()) {
      finishReason = choice.finish_reason.value();
    }
  }

  /** Build the aggregated response. Caller is responsible for postProcess. */
  tt::domain::llm::LLMResponse build(const ResponseWriterParams& params) const {
    tt::domain::llm::LLMResponse response{params.taskId};
    response.id = params.completionId;
    response.model = params.model;
    response.created = params.created;

    tt::domain::llm::LLMChoice choice;
    choice.index = 0;
    choice.text = answer.str();
    choice.reasoning = hasReasoning
                           ? std::optional<std::string>(reasoning.str())
                           : std::nullopt;
    choice.finish_reason = finishReason;
    response.choices.push_back(std::move(choice));

    response.usage = buildUsage(params);
    return response;
  }

  int tokenCount() const { return completionTokens; }

 private:
  void noteToken() {
    ++completionTokens;
    auto now = std::chrono::high_resolution_clock::now();
    if (!firstTokenTime.has_value()) {
      firstTokenTime = now;
    } else if (completionTokens == 2 && !secondTokenTime.has_value()) {
      secondTokenTime = now;
    }
  }

  tt::domain::llm::CompletionUsage buildUsage(
      const ResponseWriterParams& params) const {
    const int totalTokens = params.promptTokenCount + completionTokens;
    tt::domain::llm::CompletionUsage usage{params.promptTokenCount,
                                            completionTokens,
                                            totalTokens,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt};

    if (firstTokenTime.has_value()) {
      auto ttftUs = std::chrono::duration_cast<std::chrono::microseconds>(
          firstTokenTime.value() - startTime);
      usage.ttft_ms =
          std::round(static_cast<double>(ttftUs.count()) / 10.0) / 100.0;
    }

    if (completionTokens > 1 && firstTokenTime.has_value()) {
      auto finalTime = std::chrono::high_resolution_clock::now();
      auto baseTime = secondTokenTime.value_or(firstTokenTime.value());
      auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(
          finalTime - baseTime);
      if (totalUs.count() > 0) {
        auto secs = static_cast<double>(totalUs.count()) / 1000000.0;
        usage.tps = std::round((completionTokens - 1) / secs * 1000.0) / 1000.0;
      }
    }

    if (params.sessionId.has_value()) {
      usage.sessionId = params.sessionId;
    }
    return usage;
  }

  std::ostringstream answer;
  std::ostringstream reasoning;
  bool hasReasoning = false;
  std::string finishReason = "stop";
  int completionTokens = 0;

  std::chrono::high_resolution_clock::time_point startTime =
      std::chrono::high_resolution_clock::now();
  std::optional<std::chrono::high_resolution_clock::time_point> firstTokenTime;
  std::optional<std::chrono::high_resolution_clock::time_point> secondTokenTime;
};

}  // namespace tt::api
