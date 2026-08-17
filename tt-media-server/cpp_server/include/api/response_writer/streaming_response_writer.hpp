// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "api/response_writer/response_writer.hpp"
#include "api/stream_event_formatter.hpp"
#include "utils/concurrent_queue.hpp"

namespace tt::api {

/**
 * Response writer that delivers tokens as Server-Sent Events to the HTTP
 * client.
 *
 * Sends an OpenAI-compatible chunked stream whose wire format is decided by a
 * `StreamEventFormatter` strategy. Defaults to `ChatCompletionEventFormatter`
 * (chat.completion.chunk + `data: [DONE]`); `/v1/responses` passes a
 * `ResponsesEventFormatter` to emit `response.created`, ...,
 * `response.completed` events instead. Forwards client-disconnect detection
 * back to the LLM/disaggregation services via abort callbacks.
 */
class StreamingResponseWriter : public ResponseWriter {
 public:
  /**
   * Default factory: chat-completion SSE with optional usage chunk.
   * Equivalent to passing a `ChatCompletionEventFormatter`.
   */
  static std::shared_ptr<StreamingResponseWriter> create(
      trantor::EventLoop* loop, ResponseWriterParams params, bool includeUsage);

  /**
   * Strategy factory: caller supplies the SSE formatter (chat-completion vs
   * Responses API). When `formatter` is null, falls back to
   * `ChatCompletionEventFormatter`.
   */
  static std::shared_ptr<StreamingResponseWriter> create(
      trantor::EventLoop* loop, ResponseWriterParams params, bool includeUsage,
      std::shared_ptr<StreamEventFormatter> formatter);

  void handleTokenChunk(const LLMStreamChunk& chunk) override;
  void finalize() override;

  /**
   * Cancel the underlying request and tear down the stream. Called when the
   * client disconnects mid-stream (drogon send() returns false on a token or
   * heartbeat write).
   */
  void abort();

  /** Build the streaming HTTP response (text/event-stream, async). */
  drogon::HttpResponsePtr buildResponse();

 private:
  StreamingResponseWriter(trantor::EventLoop* loop, ResponseWriterParams params,
                          bool includeUsage,
                          std::shared_ptr<StreamEventFormatter> formatter);

  void sendSse(const std::string& sse,
               std::function<void()> onDisconnect = nullptr);
  void flushAccumulated();
  void startHeartbeat();
  void stopHeartbeat();

  trantor::EventLoop* loop;
  bool includeUsage;
  std::shared_ptr<StreamEventFormatter> formatter;

  std::shared_ptr<drogon::ResponseStreamPtr> streamPtr =
      std::make_shared<drogon::ResponseStreamPtr>();
  std::shared_ptr<std::vector<std::string>> earlyBuffer =
      std::make_shared<std::vector<std::string>>();
  std::shared_ptr<tt::utils::ConcurrentQueue<std::string>> sseBatchQueue;
  trantor::TimerId heartbeatTimerId = trantor::InvalidTimerId;

  std::atomic<bool> firstContentChunk{true};
  std::string accumulatedText;
  std::optional<std::string> lastFinishReason;
};

}  // namespace tt::api
