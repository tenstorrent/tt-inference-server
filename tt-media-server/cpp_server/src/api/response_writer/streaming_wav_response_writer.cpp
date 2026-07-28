// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/response_writer/streaming_wav_response_writer.hpp"

#include <utility>
#include <variant>

#include "utils/audio_codec.hpp"
#include "utils/logger.hpp"

namespace tt::api {

namespace {

using tt::domain::tts::TtsAudioChunk;

}  // namespace

StreamingWavResponseWriter::StreamingWavResponseWriter(
    trantor::EventLoop* loop, StreamingWavResponseWriterParams params)
    : loop(loop), params(std::move(params)) {}

std::shared_ptr<StreamingWavResponseWriter> StreamingWavResponseWriter::create(
    trantor::EventLoop* loop, StreamingWavResponseWriterParams params) {
  return std::shared_ptr<StreamingWavResponseWriter>(
      new StreamingWavResponseWriter(loop, std::move(params)));
}

void StreamingWavResponseWriter::handleEvent(
    const tt::domain::tts::TtsEvent& event) {
  if (auto chunk = std::get_if<TtsAudioChunk>(&event)) {
    handleAudioChunk(*chunk);
    return;
  }
  if (std::holds_alternative<tt::domain::tts::TtsFinishReason>(event)) {
    finalize();
  }
}

void StreamingWavResponseWriter::handleAudioChunk(const TtsAudioChunk& chunk) {
  if (done.load()) {
    return;
  }
  sendBytes(tt::utils::audio_codec::audioChunkToPcm16Bytes(chunk));
}

void StreamingWavResponseWriter::sendBytes(std::string bytes) {
  auto self = shared_from_this();
  loop->queueInLoop([self, bytes = std::move(bytes)]() {
    if (self->done.load()) {
      return;
    }
    if (*self->streamPtr) {
      if (!(*self->streamPtr)->send(bytes)) {
        self->abort();
      }
      return;
    }
    self->earlyBuffer->push_back(bytes);
  });
}

void StreamingWavResponseWriter::finalize() {
  auto self = shared_from_this();
  loop->queueInLoop([self]() {
    if (!self->done.exchange(true) && *self->streamPtr) {
      (*self->streamPtr)->close();
    }
  });
}

void StreamingWavResponseWriter::abort() {
  if (done.exchange(true)) {
    return;
  }
  TT_LOG_INFO("[StreamingWavResponseWriter] Aborting TTS stream for task {}; "
              "notifying service",
              params.task_id);
  if (params.onCancelRequest) {
    params.onCancelRequest(params.task_id);
  }
  auto self = shared_from_this();
  loop->queueInLoop([self]() {
    if (*self->streamPtr) {
      (*self->streamPtr)->close();
    }
  });
}

drogon::HttpResponsePtr StreamingWavResponseWriter::buildResponse() {
  auto self = shared_from_this();
  auto resp = drogon::HttpResponse::newAsyncStreamResponse(
      [self](drogon::ResponseStreamPtr stream) mutable {
        *self->streamPtr = std::move(stream);
        if (!(*self->streamPtr)
                 ->send(tt::utils::audio_codec::makeStreamingPcm16WavHeader(
                     self->params.sampleRateHz, self->params.channels))) {
          self->abort();
          return;
        }
        for (const auto& chunk : *self->earlyBuffer) {
          if (!(*self->streamPtr)->send(chunk)) {
            self->abort();
            return;
          }
        }
        self->earlyBuffer->clear();
        if (self->done.load()) {
          (*self->streamPtr)->close();
        }
      });

  resp->setContentTypeString("audio/wav");
  resp->addHeader("Cache-Control", "no-cache");
  resp->addHeader("Connection", "keep-alive");
  resp->addHeader("X-Accel-Buffering", "no");
  return resp;
}

}  // namespace tt::api
