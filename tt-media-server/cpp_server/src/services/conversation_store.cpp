// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/conversation_store.hpp"

#include <json/json.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "utils/logger.hpp"

namespace tt::services {

ConversationStore::ConversationStore(std::string logDir)
    : logDir(std::move(logDir)) {
  try {
    std::filesystem::create_directories(logDir);
    TT_LOG_INFO("[ConversationStore] Log directory: {}", logDir);
  } catch (const std::exception& e) {
    TT_LOG_WARN("[ConversationStore] Failed to create log dir {}: {}", logDir,
                e.what());
  }
  writerThread = std::thread([this] { writerLoop(); });
}

ConversationStore::~ConversationStore() {
  stopped.store(true, std::memory_order_relaxed);
  if (writerThread.joinable()) {
    writerThread.join();
  }
}

void ConversationStore::recordTurn(const std::string& sessionId,
                                   TurnRecord record) {
  writeQueue.push(WriteTask{sessionId, std::move(record)});
}

std::optional<std::string> ConversationStore::exportSession(
    const std::string& sessionId) const {
  auto path = logFilePath(sessionId);
  std::ifstream file(path);
  if (!file.is_open()) {
    return std::nullopt;
  }

  Json::Value turns(Json::arrayValue);
  std::string line;
  Json::CharReaderBuilder builder;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    Json::Value turn;
    std::istringstream ss(line);
    std::string errs;
    if (Json::parseFromStream(builder, ss, &turn, &errs)) {
      turns.append(std::move(turn));
    } else {
      TT_LOG_WARN("[ConversationStore] Failed to parse line for session {}: {}",
                  sessionId, errs);
    }
  }

  Json::StreamWriterBuilder writer;
  writer["indentation"] = "  ";
  return Json::writeString(writer, turns);
}

void ConversationStore::writerLoop() {
  while (!stopped.load(std::memory_order_relaxed)) {
    auto tasks = writeQueue.drain();
    if (tasks.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    for (const auto& task : tasks) {
      writeTurnToFile(task.sessionId, task.record);
    }
  }
  for (const auto& task : writeQueue.drain()) {
    writeTurnToFile(task.sessionId, task.record);
  }
}

void ConversationStore::writeTurnToFile(const std::string& sessionId,
                                        const TurnRecord& record) const {
  auto path = logFilePath(sessionId);
  std::ofstream file(path, std::ios::app);
  if (!file.is_open()) {
    TT_LOG_WARN("[ConversationStore] Failed to open log file: {}", path);
    return;
  }
  file << serializeTurn(record) << "\n";
  TT_LOG_DEBUG("[ConversationStore] Wrote turn for session {}", sessionId);
}

std::string ConversationStore::serializeTurn(const TurnRecord& record) {
  Json::Value json;
  json["timestamp_ms"] = static_cast<Json::Int64>(record.timestampMs);
  json["input_messages"] = record.inputMessages;
  json["output_text"] = record.outputText;
  json["prompt_tokens"] = record.promptTokens;
  json["completion_tokens"] = record.completionTokens;
  json["finish_reason"] = record.finishReason;
  if (record.ttftMs.has_value()) json["ttft_ms"] = record.ttftMs.value();
  if (record.tps.has_value()) json["tps"] = record.tps.value();

  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  return Json::writeString(writer, json);
}

std::string ConversationStore::logFilePath(const std::string& sessionId) const {
  return logDir + "/" + sessionId + ".jsonl";
}

}  // namespace tt::services
