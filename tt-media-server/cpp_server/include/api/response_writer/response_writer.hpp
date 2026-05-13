// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "domain/llm/llm_response.hpp"
#include "services/llm_service.hpp"
#include "services/slot_lease.hpp"

namespace tt::api {

using namespace tt::domain::llm;

/**
 * Parameters shared by every chat-completion response writer (streaming or
 * non-streaming). Wire-format-specific options live on the concrete writer.
 */
struct ResponseWriterParams {
  std::string completionId;
  std::string model;
  int64_t created;
  int promptTokenCount;
  int cachedTokenCount = 0;
  std::optional<std::string> sessionId;
  uint32_t taskId;
  std::shared_ptr<services::LLMService> service;

  // RAII handle for the session's in-flight grant. The writer holds the
  // lease for the duration of the response; the destructor releases the
  // slot. Concrete writers also call `lease.release()` explicitly on the
  // first done-transition (finalize / sendError / abort) so the slot is
  // freed before the writer's shared_ptr keepalive expires.
  services::SlotLease lease;
};

/**
 * Abstract base for chat-completion response writers. Owns the bits that are
 * truly shared between the two delivery formats:
 *  - the immutable request/session params,
 *  - the token timing state used to compute TTFT/TPS,
 *  - the idempotent done flag,
 *  - session in-flight release.
 *
 * Concrete subclasses (StreamingResponseWriter, NonStreamResponseWriter)
 * implement the wire format by overriding handleTokenChunk and finalize. The
 * controller can therefore drive both with the same streaming callback shape.
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
   * that contributes to the final response. Automatically tracks reasoning
   * tokens if the choice contains reasoning content. Returns the new token
   * count.
   */
  int noteToken(const LLMChoice& choice);

  /** Compute usage from the current accumulator state. */
  CompletionUsage buildUsage() const;

  ResponseWriterParams params;
  std::chrono::high_resolution_clock::time_point startTime =
      std::chrono::high_resolution_clock::now();
  std::optional<std::chrono::high_resolution_clock::time_point> firstTokenTime;
  std::optional<std::chrono::high_resolution_clock::time_point> secondTokenTime;
  std::atomic<int> completionTokens{0};
  std::atomic<int> reasoningTokens{0};
  std::atomic<uint32_t> specAccepts{0};
  std::atomic<uint32_t> specRejects{0};
  std::atomic<bool> done{false};
};

}  // namespace tt::api
