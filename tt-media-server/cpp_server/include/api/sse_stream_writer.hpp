// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <drogon/drogon.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "domain/llm_response.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "utils/concurrent_queue.hpp"

namespace tt::api {

struct StreamParams {
  std::string completionId;
  std::string model;
  int64_t created;
  bool includeUsage;
  bool continuousUsage;
  int promptTokensCount;
  std::optional<std::string> sessionId;
  uint32_t taskId;
  std::shared_ptr<services::LLMService> service;
  std::shared_ptr<services::SessionManager> sessionManager;
};

/**
 * Encapsulates SSE streaming infrastructure for chat completions.
 *
 * Manages the Drogon async response stream, optional token batching,
 * timing metrics, and final usage emission. Thread-safe: token callbacks
 * may arrive from service worker threads while the event loop drives
 * the HTTP response.
 */
class SseStreamWriter : public std::enable_shared_from_this<SseStreamWriter> {
 public:
  static std::shared_ptr<SseStreamWriter> create(trantor::EventLoop* loop,
                                                 StreamParams params);

  SseStreamWriter(const SseStreamWriter&) = delete;
  SseStreamWriter& operator=(const SseStreamWriter&) = delete;

  void handleTokenChunk(const domain::LLMStreamChunk& chunk);
  void finalizeStream();
  void abort();
  bool isDone() const { return done_.load(); }

  drogon::HttpResponsePtr buildResponse();

 private:
  explicit SseStreamWriter(trantor::EventLoop* loop, StreamParams params);

  void sendSse(const std::string& sse,
               std::function<void()> onDisconnect = nullptr);
  void flushAccumulated();
  domain::CompletionUsage buildFinalUsage() const;

  trantor::EventLoop* loop_;
  std::shared_ptr<drogon::ResponseStreamPtr> stream_ptr_ =
      std::make_shared<drogon::ResponseStreamPtr>();
  std::shared_ptr<std::vector<std::string>> early_buffer_ =
      std::make_shared<std::vector<std::string>>();
  std::shared_ptr<ConcurrentQueue<std::string>> accumulator_;
  std::atomic<bool> done_{false};

  std::atomic<int> completion_tokens_{0};
  std::chrono::high_resolution_clock::time_point start_time_ =
      std::chrono::high_resolution_clock::now();
  std::optional<std::chrono::high_resolution_clock::time_point>
      first_token_time_;
  std::optional<std::chrono::high_resolution_clock::time_point>
      second_token_time_;
  std::atomic<bool> first_content_chunk_{true};

  StreamParams params_;
};

}  // namespace tt::api
