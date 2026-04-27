// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "domain/llm_response.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"

namespace tt::api {

/**
 * Parameters shared by every chat-completion stream sink (SSE writer or
 * accumulating builder). Wire-format-specific options live on the concrete
 * sink instead.
 */
struct StreamSinkParams {
  std::string completionId;
  std::string model;
  int64_t created;
  int promptTokensCount;
  std::optional<std::string> sessionId;
  uint32_t taskId;
  std::shared_ptr<services::LLMService> service;
  std::shared_ptr<services::SessionManager> sessionManager;
};

/**
 * Abstract base for chat-completion stream sinks. Owns the bits that are
 * truly shared between the two delivery formats:
 *  - the immutable request/session params,
 *  - the token timing state used to compute TTFT/TPS,
 *  - the idempotent done flag,
 *  - session in-flight release.
 *
 * Concrete subclasses (SseStreamWriter, AccumulatingResponseBuilder) implement
 * the wire format by overriding handleTokenChunk and finalize. The controller
 * can therefore drive both with the same streaming callback shape.
 */
class StreamSink : public std::enable_shared_from_this<StreamSink> {
 public:
  virtual ~StreamSink() = default;

  StreamSink(const StreamSink&) = delete;
  StreamSink& operator=(const StreamSink&) = delete;

  /** Consume a single LLMStreamChunk produced by the streaming generator. */
  virtual void handleTokenChunk(const domain::LLMStreamChunk& chunk) = 0;

  /** Signal end-of-stream. Idempotent; releases in-flight slot. */
  virtual void finalize() = 0;

  bool isDone() const { return done.load(); }

 protected:
  explicit StreamSink(StreamSinkParams params);

  /**
   * Increment the completion-token counter and stamp first/second-token
   * times. Subclasses must call this from handleTokenChunk on every token
   * that contributes to the final response. Returns the new token count.
   */
  int noteToken();

  /** Compute usage from the current accumulator state. */
  domain::CompletionUsage buildUsage() const;

  /** Release the session in-flight slot if a session is associated. */
  void releaseInFlight();

  StreamSinkParams params;
  std::chrono::high_resolution_clock::time_point startTime =
      std::chrono::high_resolution_clock::now();
  std::optional<std::chrono::high_resolution_clock::time_point> firstTokenTime;
  std::optional<std::chrono::high_resolution_clock::time_point> secondTokenTime;
  std::atomic<int> completionTokens{0};
  std::atomic<bool> done{false};
};

}  // namespace tt::api
