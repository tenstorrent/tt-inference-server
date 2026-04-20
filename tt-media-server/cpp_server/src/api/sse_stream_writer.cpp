// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/sse_stream_writer.hpp"

#include <cmath>

#include "config/settings.hpp"
#include "domain/chat_completion_response.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/logger.hpp"

namespace tt::api {

SseStreamWriter::SseStreamWriter(trantor::EventLoop* loop, StreamParams params)
    : loop_(loop), params_(std::move(params)) {
  if (config::enableAccumulatedStreaming()) {
    accumulator_ = std::make_shared<tt::utils::ConcurrentQueue<std::string>>();
  }
}

std::shared_ptr<SseStreamWriter> SseStreamWriter::create(
    trantor::EventLoop* loop, StreamParams params) {
  return std::shared_ptr<SseStreamWriter>(
      new SseStreamWriter(loop, std::move(params)));
}

void SseStreamWriter::sendSse(const std::string& sse,
                              std::function<void()> onDisconnect) {
  if (!accumulator_) {
    loop_->queueInLoop([streamPtr = stream_ptr_, earlyBuffer = early_buffer_,
                        sse, onDisconnect = std::move(onDisconnect)]() {
      if (*streamPtr) {
        bool ok = (*streamPtr)->send(sse);
        if (!ok && onDisconnect) onDisconnect();
      } else if (earlyBuffer) {
        earlyBuffer->push_back(sse);
      }
    });
    return;
  }
  accumulator_->push(sse);
  if (accumulator_->size() >= config::maxAccumulatedTokens()) {
    auto accumulated = accumulator_->drain();
    std::string batch;
    for (auto& s : accumulated) batch.append(s);
    loop_->queueInLoop([streamPtr = stream_ptr_, earlyBuffer = early_buffer_,
                        batch = std::move(batch),
                        onDisconnect = std::move(onDisconnect)]() {
      if (*streamPtr) {
        bool ok = (*streamPtr)->send(batch);
        if (!ok && onDisconnect) onDisconnect();
      } else if (earlyBuffer) {
        earlyBuffer->push_back(batch);
      }
    });
  }
}

void SseStreamWriter::flushAccumulated() {
  if (!accumulator_) return;
  auto accumulated = accumulator_->drain();
  if (!accumulated.empty()) {
    std::string batch;
    for (auto& s : accumulated) batch.append(s);
    if (*stream_ptr_) (*stream_ptr_)->send(batch);
  }
}

domain::CompletionUsage SseStreamWriter::buildFinalUsage() const {
  const int completionTokens = completion_tokens_.load();
  const int totalTokens = params_.promptTokensCount + completionTokens;

  domain::CompletionUsage usage{params_.promptTokensCount,
                                completionTokens,
                                totalTokens,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt};

  if (first_token_time_.has_value()) {
    auto ttftUs = std::chrono::duration_cast<std::chrono::microseconds>(
        first_token_time_.value() - start_time_);
    usage.ttft_ms =
        std::round(static_cast<double>(ttftUs.count()) / 10.0) / 100.0;
  }

  if (completionTokens > 1 && first_token_time_.has_value()) {
    auto finalTime = std::chrono::high_resolution_clock::now();
    auto baseTime = second_token_time_.value_or(first_token_time_.value());
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(
        finalTime - baseTime);
    if (totalUs.count() > 0) {
      auto secs = static_cast<double>(totalUs.count()) / 1000000.0;
      usage.tps = std::round((completionTokens - 1) / secs * 1000.0) / 1000.0;
    }
  }

  if (params_.sessionId.has_value()) {
    usage.sessionId = params_.sessionId;
  }
  return usage;
}

void SseStreamWriter::handleTokenChunk(const domain::LLMStreamChunk& chunk) {
  if (done_.load()) return;

  const int currentTokens = completion_tokens_.fetch_add(1) + 1;

  auto now = std::chrono::high_resolution_clock::now();
  if (!first_token_time_.has_value()) {
    first_token_time_ = now;
  } else if (currentTokens == 2 && !second_token_time_.has_value()) {
    second_token_time_ = now;
  }

  std::optional<domain::CompletionUsage> usage;
  if (params_.continuousUsage) {
    usage = domain::CompletionUsage{params_.promptTokensCount,
                                    currentTokens,
                                    params_.promptTokensCount + currentTokens,
                                    std::nullopt,
                                    std::nullopt,
                                    params_.sessionId};
  }

  auto streamChunk = domain::ChatCompletionStreamChunk::makeContentChunk(
      params_.completionId, params_.model, params_.created, chunk.choices[0],
      usage);

  std::string sse;
  if (first_content_chunk_.exchange(false)) {
    std::optional<domain::CompletionUsage> initialUsage;
    if (params_.continuousUsage) {
      initialUsage = domain::CompletionUsage{
          params_.promptTokensCount, 0, 0, std::nullopt, std::nullopt,
          params_.sessionId};
    }
    auto initialChunk = domain::ChatCompletionStreamChunk::makeInitialChunk(
        params_.completionId, params_.model, params_.created, initialUsage);
    sse = initialChunk.toSSE() + streamChunk.toSSE();
  } else {
    sse = streamChunk.toSSE();
  }

  if (!sse.empty()) {
    auto self = shared_from_this();
    sendSse(sse, [self]() { self->abort(); });
  }
}

void SseStreamWriter::finalizeStream() {
  auto self = shared_from_this();
  loop_->queueInLoop([self]() {
    if (!self->done_.exchange(true) && *self->stream_ptr_) {
      self->flushAccumulated();

      if (self->params_.includeUsage) {
        auto usage = self->buildFinalUsage();
        (*self->stream_ptr_)
            ->send(domain::ChatCompletionStreamChunk::makeUsageChunk(
                       self->params_.completionId, self->params_.model,
                       self->params_.created, usage)
                       .toSSE());
      }

      (*self->stream_ptr_)->send("data: [DONE]\n\n");
      (*self->stream_ptr_)->close();

      if (self->params_.sessionId.has_value() && self->params_.sessionManager) {
        self->params_.sessionManager->setSessionInFlight(
            self->params_.sessionId.value(), false);
      }
    }
  });
}

void SseStreamWriter::abort() {
  if (!done_.exchange(true)) {
    TT_LOG_INFO("[SseStreamWriter] Client disconnected, aborting task {}",
                params_.taskId);
    params_.service->abortRequest(params_.taskId);
    if (params_.sessionId.has_value() && params_.sessionManager) {
      params_.sessionManager->setSessionInFlight(params_.sessionId.value(), false);
    }
  }
}

drogon::HttpResponsePtr SseStreamWriter::buildResponse() {
  auto self = shared_from_this();
  auto resp = drogon::HttpResponse::newAsyncStreamResponse(
      [self](drogon::ResponseStreamPtr stream) mutable {
        *self->stream_ptr_ = std::move(stream);
        for (const auto& event : *self->early_buffer_) {
          (*self->stream_ptr_)->send(event);
        }
        self->early_buffer_->clear();
        if (self->done_.load()) {
          (*self->stream_ptr_)->close();
        }
      });

  resp->setContentTypeString("text/event-stream");
  resp->addHeader("Cache-Control", "no-cache");
  resp->addHeader("Connection", "keep-alive");
  resp->addHeader("X-Accel-Buffering", "no");

  return resp;
}

}  // namespace tt::api
