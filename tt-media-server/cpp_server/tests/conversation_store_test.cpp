// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/conversation_store.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace {

tt::services::TurnRecord makeTurn(const std::string& userContent,
                                  const std::string& outputText, double ttftMs,
                                  double tps, int promptTokens,
                                  int completionTokens,
                                  const std::string& finishReason = "stop") {
  tt::services::TurnRecord record;

  Json::Value messages(Json::arrayValue);
  Json::Value msg;
  msg["role"] = "user";
  msg["content"] = userContent;
  messages.append(msg);

  record.inputMessages = messages;
  record.outputText = outputText;
  record.ttftMs = ttftMs;
  record.tps = tps;
  record.promptTokens = promptTokens;
  record.completionTokens = completionTokens;
  record.finishReason = finishReason;
  record.timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  return record;
}

// Waits up to maxWaitMs for the .jsonl file for sessionId to contain at least
// expectedTurns lines. Returns true if the condition is met in time.
bool waitForFile(const std::string& logDir, const std::string& sessionId,
                 int expectedTurns, int maxWaitMs = 500) {
  auto path = logDir + "/" + sessionId + ".jsonl";
  for (int waited = 0; waited < maxWaitMs; waited += 10) {
    std::ifstream f(path);
    if (f.is_open()) {
      int lines = 0;
      std::string line;
      while (std::getline(f, line)) {
        if (!line.empty()) ++lines;
      }
      if (lines >= expectedTurns) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

class ConversationStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmpDir = std::filesystem::temp_directory_path() /
             ("conversation_store_test_" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(tmpDir);
  }

  void TearDown() override { std::filesystem::remove_all(tmpDir); }

  std::string logDir() const { return tmpDir.string(); }

  std::filesystem::path tmpDir;
};

TEST_F(ConversationStoreTest, ExportUnknownSessionReturnsNullopt) {
  tt::services::ConversationStore store(logDir());
  auto result = store.exportSession("nonexistent-session-id");
  EXPECT_FALSE(result.has_value());
}

TEST_F(ConversationStoreTest, RecordAndExportSingleTurn) {
  tt::services::ConversationStore store(logDir());

  const std::string sessionId = "test-session-001";
  auto turn =
      makeTurn("Hello, what is 2+2?", "The answer is 4.", 8.3, 120.5, 12, 5);

  store.recordTurn(sessionId, turn);

  ASSERT_TRUE(waitForFile(logDir(), sessionId, 1))
      << "Timed out waiting for turn to be written";

  auto exported = store.exportSession(sessionId);
  ASSERT_TRUE(exported.has_value());

  Json::Value turns;
  Json::CharReaderBuilder builder;
  std::istringstream ss(exported.value());
  std::string errs;
  ASSERT_TRUE(Json::parseFromStream(builder, ss, &turns, &errs)) << errs;

  ASSERT_EQ(turns.size(), 1u);

  const Json::Value& t = turns[0];
  EXPECT_EQ(t["output_text"].asString(), "The answer is 4.");
  EXPECT_EQ(t["finish_reason"].asString(), "stop");
  EXPECT_EQ(t["prompt_tokens"].asInt(), 12);
  EXPECT_EQ(t["completion_tokens"].asInt(), 5);
  EXPECT_DOUBLE_EQ(t["ttft_ms"].asDouble(), 8.3);
  EXPECT_DOUBLE_EQ(t["tps"].asDouble(), 120.5);

  ASSERT_TRUE(t["input_messages"].isArray());
  ASSERT_EQ(t["input_messages"].size(), 1u);
  EXPECT_EQ(t["input_messages"][0]["role"].asString(), "user");
  EXPECT_EQ(t["input_messages"][0]["content"].asString(),
            "Hello, what is 2+2?");
}

TEST_F(ConversationStoreTest, MultiTurnOrderPreserved) {
  tt::services::ConversationStore store(logDir());

  const std::string sessionId = "test-session-002";
  store.recordTurn(sessionId,
                   makeTurn("Turn 1", "Response 1", 8.0, 100.0, 5, 3));
  store.recordTurn(sessionId,
                   makeTurn("Turn 2", "Response 2", 9.0, 110.0, 6, 4));
  store.recordTurn(sessionId,
                   makeTurn("Turn 3", "Response 3", 7.5, 105.0, 4, 2));

  ASSERT_TRUE(waitForFile(logDir(), sessionId, 3))
      << "Timed out waiting for 3 turns to be written";

  auto exported = store.exportSession(sessionId);
  ASSERT_TRUE(exported.has_value());

  Json::Value turns;
  Json::CharReaderBuilder builder;
  std::istringstream ss(exported.value());
  std::string errs;
  ASSERT_TRUE(Json::parseFromStream(builder, ss, &turns, &errs)) << errs;

  ASSERT_EQ(turns.size(), 3u);
  EXPECT_EQ(turns[0]["output_text"].asString(), "Response 1");
  EXPECT_EQ(turns[1]["output_text"].asString(), "Response 2");
  EXPECT_EQ(turns[2]["output_text"].asString(), "Response 3");
}

TEST_F(ConversationStoreTest, MultipleSessionsAreIsolated) {
  tt::services::ConversationStore store(logDir());

  const std::string sessionA = "session-aaa";
  const std::string sessionB = "session-bbb";

  store.recordTurn(sessionA,
                   makeTurn("Question A", "Answer A", 8.0, 100.0, 5, 3));
  store.recordTurn(sessionB,
                   makeTurn("Question B", "Answer B", 9.0, 110.0, 6, 4));

  ASSERT_TRUE(waitForFile(logDir(), sessionA, 1));
  ASSERT_TRUE(waitForFile(logDir(), sessionB, 1));

  auto exportedA = store.exportSession(sessionA);
  auto exportedB = store.exportSession(sessionB);

  ASSERT_TRUE(exportedA.has_value());
  ASSERT_TRUE(exportedB.has_value());

  Json::Value turnsA, turnsB;
  Json::CharReaderBuilder builder;
  std::string errs;

  std::istringstream ssA(exportedA.value());
  ASSERT_TRUE(Json::parseFromStream(builder, ssA, &turnsA, &errs)) << errs;

  std::istringstream ssB(exportedB.value());
  ASSERT_TRUE(Json::parseFromStream(builder, ssB, &turnsB, &errs)) << errs;

  ASSERT_EQ(turnsA.size(), 1u);
  ASSERT_EQ(turnsB.size(), 1u);
  EXPECT_EQ(turnsA[0]["output_text"].asString(), "Answer A");
  EXPECT_EQ(turnsB[0]["output_text"].asString(), "Answer B");

  // Verify separate files on disk
  EXPECT_TRUE(std::filesystem::exists(logDir() + "/" + sessionA + ".jsonl"));
  EXPECT_TRUE(std::filesystem::exists(logDir() + "/" + sessionB + ".jsonl"));
}

TEST_F(ConversationStoreTest, TurnWithOptionalFieldsMissing) {
  tt::services::ConversationStore store(logDir());

  const std::string sessionId = "test-session-004";

  tt::services::TurnRecord record;
  Json::Value messages(Json::arrayValue);
  record.inputMessages = messages;
  record.outputText = "some output";
  // ttftMs and tps deliberately left as nullopt
  record.finishReason = "length";
  record.timestampMs = 1000;

  store.recordTurn(sessionId, record);

  ASSERT_TRUE(waitForFile(logDir(), sessionId, 1));

  auto exported = store.exportSession(sessionId);
  ASSERT_TRUE(exported.has_value());

  Json::Value turns;
  Json::CharReaderBuilder builder;
  std::istringstream ss(exported.value());
  std::string errs;
  ASSERT_TRUE(Json::parseFromStream(builder, ss, &turns, &errs)) << errs;

  ASSERT_EQ(turns.size(), 1u);
  EXPECT_EQ(turns[0]["output_text"].asString(), "some output");
  EXPECT_EQ(turns[0]["finish_reason"].asString(), "length");
  EXPECT_FALSE(turns[0].isMember("ttft_ms"));
  EXPECT_FALSE(turns[0].isMember("tps"));
}

}  // namespace
