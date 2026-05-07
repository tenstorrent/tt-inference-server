// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "domain/llm/llm_response.hpp"

namespace tt::api {

using namespace tt::domain::llm;

/**
 * Parameters shared by every chat-completion response writer.
 *
 * Holds presentation-layer values only: id, model, timestamp, prompt-token
 * count, optional session id (for usage display), task id, and a
 * controller-supplied callback that releases the session in-flight slot.
 *
 * Notably does NOT hold a service or session-manager pointer: the writer must
 * stay agnostic of those layers. Cancellation is wired separately via the
 * abortFn parameter on the streaming writer.
 */
struct ResponseWriterParams {
  std::string completionId;
  std::string model;
  int64_t created;
  int promptTokenCount;
  std::optional<std::string> sessionId;
  uint32_t taskId;
  std::function<void()> releaseInFlightFn;
};

/**
 * Abstract base for chat-completion response writers. Owns the bits that are
 * truly shared between the two delivery formats:
 *  - the immutable request/session params,
 *  - the token timing state used to compute TTFT/TPS,
 *  - the idempotent done flag,
 *  - session in-flight release.
 *
 * Concrete subclasses (currently StreamingResponseWriter) implement the wire
 * format by overriding handleTokenChunk and finalize. The controller drives
 * the writer with a chunk-callback shape; the non-streaming path uses
 * LLMResponseAccumulator + a controller-side sink lambda instead.
 *
 * Thread safety: Callbacks are serialized by the LLMService consumer thread,
 * so noteToken() relies on this serialization for timing accuracy. The atomic
 * done flag protects finalize() and abort() from concurrent invocation.
 */
class ResponseWriter : public std::enable_shared_from_this<ResponseWriter> {
 public:
  virtual ~ResponseWriter() = default;

  ResponseWriter(const ResponseWriter&) = delete;
  ResponseWriter& operator=(const ResponseWriter&) = delete;

  /** Consume a single LLMStreamChunk produced by the streaming generator. */
  virtual void handleTokenChunk(const LLMStreamChunk& chunk) = 0;

  /** Signal end-of-stream. Idempotent; releases in-flight slot. */
  virtual void finalize() = 0;

  bool isDone() const { return done.load(); }

 protected:
  explicit ResponseWriter(ResponseWriterParams params);

  /**
   * Increment the completion-token counter and stamp first/second-token
   * times. Subclasses must call this from handleTokenChunk on every token
   * that contributes to the final response. Returns the new token count.
   */
  int noteToken();

  /** Compute usage from the current accumulator state. */
  CompletionUsage buildUsage() const;

  /** Release the session in-flight slot if a session is associated. */
  void releaseInFlight();

  ResponseWriterParams params;
  std::chrono::high_resolution_clock::time_point startTime =
      std::chrono::high_resolution_clock::now();
  std::optional<std::chrono::high_resolution_clock::time_point> firstTokenTime;
  std::optional<std::chrono::high_resolution_clock::time_point> secondTokenTime;
  std::atomic<int> completionTokens{0};
  std::atomic<bool> done{false};
};

}  // namespace tt::api
