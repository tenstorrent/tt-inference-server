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
#include "utils/concurrent_queue.hpp"

namespace tt::api {

/**
 * Response writer that delivers tokens as Server-Sent Events to the HTTP
 * client.
 *
 * Sends an OpenAI-compatible chunked stream: optional initial role-only
 * chunk, one delta per token (optionally batched via the accumulated-
 * streaming config), an optional final usage chunk, and a "data: [DONE]\n\n"
 * terminator. Forwards client-disconnect detection back to the LLM service
 * via abortRequest.
 */
class StreamingResponseWriter : public ResponseWriter {
 public:
  static std::shared_ptr<StreamingResponseWriter> create(
      trantor::EventLoop* loop, ResponseWriterParams params, bool includeUsage,
      bool continuousUsage);

  void handleTokenChunk(const domain::LLMStreamChunk& chunk) override;
  void finalize() override;

  /**
   * Cancel the underlying request and tear down the stream. Called when the
   * client disconnects mid-stream (drogon send() returns false).
   */
  void abort();

  /** Build the streaming HTTP response (text/event-stream, async). */
  drogon::HttpResponsePtr buildResponse();

 private:
  StreamingResponseWriter(trantor::EventLoop* loop, ResponseWriterParams params,
                          bool includeUsage, bool continuousUsage);

  void sendSse(const std::string& sse,
               std::function<void()> onDisconnect = nullptr);
  void flushAccumulated();

  trantor::EventLoop* loop;
  bool includeUsage;
  bool continuousUsage;

  std::shared_ptr<drogon::ResponseStreamPtr> streamPtr =
      std::make_shared<drogon::ResponseStreamPtr>();
  std::shared_ptr<std::vector<std::string>> earlyBuffer =
      std::make_shared<std::vector<std::string>>();
  std::shared_ptr<tt::utils::ConcurrentQueue<std::string>> sseBatchQueue;

  std::atomic<bool> firstContentChunk{true};
};

}  // namespace tt::api
