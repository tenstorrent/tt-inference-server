// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/response_writer/streaming_response_writer.hpp"

#include <utility>

#include "config/settings.hpp"
#include "domain/chat_completion_response.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/logger.hpp"

namespace tt::api {

StreamingResponseWriter::StreamingResponseWriter(trantor::EventLoop* loop,
                                                 ResponseWriterParams params,
                                                 bool includeUsage,
                                                 bool continuousUsage)
    : ResponseWriter(std::move(params)),
      loop(loop),
      includeUsage(includeUsage),
      continuousUsage(continuousUsage) {
  if (config::enableAccumulatedStreaming()) {
    sseBatchQueue = std::make_shared<tt::utils::ConcurrentQueue<std::string>>();
  }
}

std::shared_ptr<StreamingResponseWriter> StreamingResponseWriter::create(
    trantor::EventLoop* loop, ResponseWriterParams params, bool includeUsage,
    bool continuousUsage) {
  return std::shared_ptr<StreamingResponseWriter>(new StreamingResponseWriter(
      loop, std::move(params), includeUsage, continuousUsage));
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

void StreamingResponseWriter::handleTokenChunk(
    const domain::LLMStreamChunk& chunk) {
  if (done.load()) return;
  if (chunk.choices.empty()) return;

  const int currentTokens = noteToken();

  std::optional<domain::CompletionUsage> usage;
  if (continuousUsage) {
    usage = domain::CompletionUsage{params.promptTokenCount,
                                    currentTokens,
                                    params.promptTokenCount + currentTokens,
                                    std::nullopt,
                                    std::nullopt,
                                    params.sessionId};
  }

  auto streamChunk = domain::ChatCompletionStreamChunk::makeContentChunk(
      params.completionId, params.model, params.created, chunk.choices[0],
      usage);

  std::string sse;
  if (firstContentChunk.exchange(false)) {
    std::optional<domain::CompletionUsage> initialUsage;
    if (continuousUsage) {
      initialUsage = domain::CompletionUsage{
          params.promptTokenCount, 0, 0, std::nullopt, std::nullopt,
          params.sessionId};
    }
    auto initialChunk = domain::ChatCompletionStreamChunk::makeInitialChunk(
        params.completionId, params.model, params.created, initialUsage);
    sse = initialChunk.toSSE() + streamChunk.toSSE();
  } else {
    sse = streamChunk.toSSE();
  }

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
    if (!self->done.exchange(true) && *self->streamPtr) {
      self->flushAccumulated();

      if (self->includeUsage) {
        auto usage = self->buildUsage();
        (*self->streamPtr)
            ->send(domain::ChatCompletionStreamChunk::makeUsageChunk(
                       self->params.completionId, self->params.model,
                       self->params.created, usage)
                       .toSSE());
      }

      (*self->streamPtr)->send("data: [DONE]\n\n");
      (*self->streamPtr)->close();

      self->releaseInFlight();
    }
  });
}

void StreamingResponseWriter::abort() {
  if (!done.exchange(true)) {
    TT_LOG_INFO(
        "[StreamingResponseWriter] Client disconnected, aborting task {}",
        params.taskId);
    if (params.service) params.service->abortRequest(params.taskId);
    releaseInFlight();
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
