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
  Json::Value input_messages;  // JSON array of {role, content} objects
  std::string output_text;
  std::optional<double> ttft_ms;
  std::optional<double> tps;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int64_t timestamp_ms = 0;
  std::string finish_reason;
};

/**
 * Stores per-session conversation turns (input + output + timing) to disk for
 * later download. Each session's turns are appended as newline-delimited JSON
 * to {logDir}/{sessionId}.jsonl by a background writer thread so the hot
 * request path is never blocked on I/O.
 *
 * Removal: delete this file, remove it from ServiceContainer, and remove the
 * two recordTurn call-sites in LLMController / SseStreamWriter.
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
  // Reads from the .jsonl file on disk.
  // Note: turns enqueued but not yet flushed by the background thread may not
  // appear immediately; in practice the writer thread drains within ~5 ms.
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

  std::string logDir_;
  utils::ConcurrentQueue<WriteTask> writeQueue_;
  std::atomic<bool> stopped_{false};
  std::thread writerThread_;
};

}  // namespace tt::services
