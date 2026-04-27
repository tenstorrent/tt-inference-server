// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/stream_sink.hpp"

#include <cmath>
#include <utility>

namespace tt::api {

StreamSink::StreamSink(StreamSinkParams params) : params(std::move(params)) {}

int StreamSink::noteToken() {
  const int current = completionTokens.fetch_add(1) + 1;
  auto now = std::chrono::high_resolution_clock::now();
  if (!firstTokenTime.has_value()) {
    firstTokenTime = now;
  } else if (current == 2 && !secondTokenTime.has_value()) {
    secondTokenTime = now;
  }
  return current;
}

domain::CompletionUsage StreamSink::buildUsage() const {
  const int tokens = completionTokens.load();
  const int totalTokens = params.promptTokensCount + tokens;
  domain::CompletionUsage usage{params.promptTokensCount,
                                tokens,
                                totalTokens,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt};

  if (firstTokenTime.has_value()) {
    auto ttftUs = std::chrono::duration_cast<std::chrono::microseconds>(
        firstTokenTime.value() - startTime);
    usage.ttft_ms =
        std::round(static_cast<double>(ttftUs.count()) / 10.0) / 100.0;
  }

  if (tokens > 1 && firstTokenTime.has_value()) {
    auto finalTime = std::chrono::high_resolution_clock::now();
    auto baseTime = secondTokenTime.value_or(firstTokenTime.value());
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(
        finalTime - baseTime);
    if (totalUs.count() > 0) {
      auto secs = static_cast<double>(totalUs.count()) / 1000000.0;
      usage.tps = std::round((tokens - 1) / secs * 1000.0) / 1000.0;
    }
  }

  if (params.sessionId.has_value()) {
    usage.sessionId = params.sessionId;
  }
  return usage;
}

void StreamSink::releaseInFlight() {
  if (params.sessionId.has_value() && params.sessionManager) {
    params.sessionManager->releaseInFlight(params.sessionId.value());
  }
}

}  // namespace tt::api
