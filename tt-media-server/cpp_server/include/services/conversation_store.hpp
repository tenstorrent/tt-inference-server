// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>

#include "utils/concurrent_queue.hpp"

namespace tt::services {

struct TurnRecord {
  Json::Value inputMessages;
  std::string outputText;
  std::optional<double> ttftMs;
  std::optional<double> tps;
  int promptTokens = 0;
  int completionTokens = 0;
  int64_t timestampMs = 0;
  std::string finishReason;
};

/**
 * Stores per-session conversation turns (input + output + timing) to disk for
 * later download. Each session's turns are appended as newline-delimited JSON
 * to {logDir}/{sessionId}.jsonl by a background writer thread so the hot
 * request path is never blocked on I/O.
 */
class ConversationStore {
 public:
  explicit ConversationStore(std::string logDir);
  ~ConversationStore();

  ConversationStore(const ConversationStore&) = delete;
  ConversationStore& operator=(const ConversationStore&) = delete;

  // Thread-safe: enqueue a completed turn for async file write.
  void recordTurn(const std::string& sessionId, TurnRecord record);

  // Return all recorded turns for the session as a JSON array string.
  // Reads from the .jsonl file on disk. Turns enqueued but not yet
  // flushed by the background thread may not appear immediately
  std::optional<std::string> exportSession(const std::string& sessionId) const;

 private:
  struct WriteTask {
    std::string sessionId;
    TurnRecord record;
  };

  void writerLoop();
  void writeTurnToFile(const std::string& sessionId,
                       const TurnRecord& record) const;
  static std::string serializeTurn(const TurnRecord& record);
  std::string logFilePath(const std::string& sessionId) const;

  std::string logDir;
  utils::ConcurrentQueue<WriteTask> writeQueue;
  std::atomic<bool> stopped{false};
  std::thread writerThread;
};

}  // namespace tt::services
