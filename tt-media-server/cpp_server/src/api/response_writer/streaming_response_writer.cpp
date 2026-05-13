// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/response_writer/streaming_response_writer.hpp"

#include <utility>

#include "config/settings.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/logger.hpp"

namespace tt::api {

using namespace tt::domain::llm;

StreamingResponseWriter::StreamingResponseWriter(
    trantor::EventLoop* loop, ResponseWriterParams params, bool includeUsage,
    std::shared_ptr<StreamEventFormatter> formatter)
    : ResponseWriter(std::move(params)),
      loop(loop),
      includeUsage(includeUsage),
      formatter(std::move(formatter)) {
  if (config::enableAccumulatedStreaming()) {
    sseBatchQueue = std::make_shared<tt::utils::ConcurrentQueue<std::string>>();
  }
}

std::shared_ptr<StreamingResponseWriter> StreamingResponseWriter::create(
    trantor::EventLoop* loop, ResponseWriterParams params, bool includeUsage) {
  return create(loop, std::move(params), includeUsage,
                std::make_shared<ChatCompletionEventFormatter>());
}

std::shared_ptr<StreamingResponseWriter> StreamingResponseWriter::create(
    trantor::EventLoop* loop, ResponseWriterParams params, bool includeUsage,
    std::shared_ptr<StreamEventFormatter> formatter) {
  if (!formatter) {
    formatter = std::make_shared<ChatCompletionEventFormatter>();
  }
  return std::shared_ptr<StreamingResponseWriter>(new StreamingResponseWriter(
      loop, std::move(params), includeUsage, std::move(formatter)));
}

void StreamingResponseWriter::sendSse(const std::string& sse,
                                      std::function<void()> onDisconnect) {
  if (!sseBatchQueue) {
    loop->queueInLoop([streamPtr = this->streamPtr,
                       earlyBuffer = this->earlyBuffer, sse,
                       onDisconnect = std::move(onDisconnect)]() {
      if (*streamPtr) {
        bool ok = (*streamPtr)->send(sse);
        if (!ok && onDisconnect) onDisconnect();
      } else if (earlyBuffer) {
        earlyBuffer->push_back(sse);
      }
    });
    return;
  }
  sseBatchQueue->push(sse);
  if (sseBatchQueue->size() >= config::maxAccumulatedTokens()) {
    auto accumulated = sseBatchQueue->drain();
    std::string batch;
    for (auto& s : accumulated) batch.append(s);
    loop->queueInLoop(
        [streamPtr = this->streamPtr, earlyBuffer = this->earlyBuffer,
         batch = std::move(batch), onDisconnect = std::move(onDisconnect)]() {
          if (*streamPtr) {
            bool ok = (*streamPtr)->send(batch);
            if (!ok && onDisconnect) onDisconnect();
          } else if (earlyBuffer) {
            earlyBuffer->push_back(batch);
          }
        });
  }
}

void StreamingResponseWriter::flushAccumulated() {
  if (!sseBatchQueue) return;
  auto accumulated = sseBatchQueue->drain();
  if (!accumulated.empty()) {
    std::string batch;
    for (auto& s : accumulated) batch.append(s);
    if (*streamPtr) (*streamPtr)->send(batch);
  }
}

void StreamingResponseWriter::handleTokenChunk(const LLMStreamChunk& chunk) {
  if (done.load()) return;
  if (chunk.choices.empty()) return;

  const auto& choice = chunk.choices[0];
  const int currentTokens = noteToken(choice);

  const std::string accumulatedSoFar = accumulatedText;
  accumulatedText += choice.text;
  if (choice.finish_reason.has_value()) {
    lastFinishReason = choice.finish_reason;
  }

  std::string sse;
  if (firstContentChunk.exchange(false)) {
    sse += formatter->formatInitialEvents(params, std::nullopt);
  }
  sse += formatter->formatTokenEvents(params, chunk, std::nullopt,
                                      currentTokens, accumulatedSoFar);

  if (!sse.empty()) {
    auto self =
        std::static_pointer_cast<StreamingResponseWriter>(shared_from_this());
    sendSse(sse, [self]() { self->abort(); });
  }
}

void StreamingResponseWriter::finalize() {
  auto self =
      std::static_pointer_cast<StreamingResponseWriter>(shared_from_this());
  loop->queueInLoop([self]() {
    if (!self->done.exchange(true)) {
      // Release the in-flight slot before flushing the final SSE -- the
      // session becomes idle the instant generation is done, regardless
      // of network drain.
      self->params.lease.release();

      if (*self->streamPtr) {
        self->flushAccumulated();

        auto usage = self->buildUsage();
        auto finalSse = self->formatter->formatFinalEvents(
            self->params, usage, self->accumulatedText, self->lastFinishReason,
            self->includeUsage);
        if (!finalSse.empty()) {
          (*self->streamPtr)->send(finalSse);
        }
        (*self->streamPtr)->close();
      }
    }
  });
}

void StreamingResponseWriter::abort() {
  if (!done.exchange(true)) {
    TT_LOG_INFO(
        "[StreamingResponseWriter] Client disconnected, aborting task {}",
        params.taskId);
    if (params.service) params.service->abortRequest(params.taskId);
    params.lease.release();
  }
}

drogon::HttpResponsePtr StreamingResponseWriter::buildResponse() {
  auto self =
      std::static_pointer_cast<StreamingResponseWriter>(shared_from_this());
  auto resp = drogon::HttpResponse::newAsyncStreamResponse(
      [self](drogon::ResponseStreamPtr stream) mutable {
        *self->streamPtr = std::move(stream);
        for (const auto& event : *self->earlyBuffer) {
          (*self->streamPtr)->send(event);
        }
        self->earlyBuffer->clear();
        if (self->done.load()) {
          (*self->streamPtr)->close();
        }
      });

  resp->setContentTypeString("text/event-stream");
  resp->addHeader("Cache-Control", "no-cache");
  resp->addHeader("Connection", "keep-alive");
  resp->addHeader("X-Accel-Buffering", "no");

  return resp;
}

}  // namespace tt::api
