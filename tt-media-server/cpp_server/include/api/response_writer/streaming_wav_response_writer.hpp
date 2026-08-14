// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "domain/tts/tts_types.hpp"

namespace tt::api {

struct StreamingWavResponseWriterParams {
  uint32_t task_id = 0;
  uint32_t sampleRateHz = 0;
  uint16_t channels = 0;
  std::function<void(uint32_t)> onCancelRequest;
};

/** Binary response writer for TTS audio/wav streams. */
class StreamingWavResponseWriter
    : public std::enable_shared_from_this<StreamingWavResponseWriter> {
 public:
  static std::shared_ptr<StreamingWavResponseWriter> create(
      trantor::EventLoop* loop, StreamingWavResponseWriterParams params);

  void handleEvent(const tt::domain::tts::TtsEvent& event);
  drogon::HttpResponsePtr buildResponse();

 private:
  StreamingWavResponseWriter(trantor::EventLoop* loop,
                             StreamingWavResponseWriterParams params);

  void handleAudioChunk(const tt::domain::tts::TtsAudioChunk& chunk);
  void sendBytes(std::string bytes);
  void finalize();
  void abort();

  trantor::EventLoop* loop;
  StreamingWavResponseWriterParams params;
  std::shared_ptr<drogon::ResponseStreamPtr> streamPtr =
      std::make_shared<drogon::ResponseStreamPtr>();
  std::shared_ptr<std::vector<std::string>> earlyBuffer =
      std::make_shared<std::vector<std::string>>();
  std::atomic<bool> done{false};
};

}  // namespace tt::api
