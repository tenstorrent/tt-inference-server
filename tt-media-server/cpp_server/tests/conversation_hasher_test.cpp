// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/conversation_hasher.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "config/settings.hpp"
#include "domain/chat_message.hpp"
#include "utils/tokenizers/tokenizer.hpp"

using namespace tt::domain;
using namespace tt::utils;
using tt::utils::tokenizers::activeTokenizer;

namespace {

ChatMessage makeMessage(std::string role, std::string content) {
  ChatMessage m;
  m.role = std::move(role);
  m.content = std::move(content);
  return m;
}

// ---------------------------------------------------------------------------
// Pure helpers (no tokenizer)
// ---------------------------------------------------------------------------

TEST(ConversationHasherLogic, StripToolMessages_DropsToolAndFunction) {
  std::vector<ChatMessage> in = {
      makeMessage("system", "sys"),   makeMessage("user", "u1"),
      makeMessage("tool", "t1"),      makeMessage("function", "f1"),
      makeMessage("assistant", "a1"),
  };
  auto out = stripToolMessages(in);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].role, "system");
  EXPECT_EQ(out[1].role, "user");
  EXPECT_EQ(out[2].role, "assistant");
}

TEST(ConversationHasherLogic, ExtractPriorTurnPrefix_RequiresUserTail) {
  EXPECT_FALSE(extractPriorTurnPrefix({}).has_value());
  EXPECT_FALSE(extractPriorTurnPrefix({makeMessage("user", "x")}).has_value());
}

TEST(ConversationHasherLogic, ExtractPriorTurnPrefix_TooShortAfterStrip) {
  std::vector<ChatMessage> oneUser = {makeMessage("user", "only")};
  EXPECT_FALSE(extractPriorTurnPrefix(oneUser).has_value());
}

TEST(ConversationHasherLogic, ExtractPriorTurnPrefix_SecondToLastNotAssistant) {
  std::vector<ChatMessage> userUser = {
      makeMessage("user", "a"),
      makeMessage("user", "b"),
  };
  EXPECT_FALSE(extractPriorTurnPrefix(userUser).has_value());
}

TEST(ConversationHasherLogic, ExtractPriorTurnPrefix_AssistantUserOnly) {
  std::vector<ChatMessage> pair = {
      makeMessage("assistant", "a"),
      makeMessage("user", "b"),
  };
  EXPECT_FALSE(extractPriorTurnPrefix(pair).has_value());
}

TEST(ConversationHasherLogic, ExtractPriorTurnPrefix_TrailingPair) {
  std::vector<ChatMessage> thread = {
      makeMessage("system", "s"),
      makeMessage("user", "first"),
      makeMessage("assistant", "mid"),
      makeMessage("user", "last"),
  };
  auto prior = extractPriorTurnPrefix(thread);
  ASSERT_TRUE(prior.has_value());
  ASSERT_EQ(prior->size(), 2u);
  EXPECT_EQ((*prior)[0].role, "system");
  EXPECT_EQ((*prior)[0].content, "s");
  EXPECT_EQ((*prior)[1].role, "user");
  EXPECT_EQ((*prior)[1].content, "first");
}

TEST(ConversationHasherLogic, ExtractPriorTurnPrefix_IgnoresToolInTail) {
  std::vector<ChatMessage> withTool = {
      makeMessage("user", "q"),
      makeMessage("assistant", "with tool next"),
      makeMessage("tool", "result"),
      makeMessage("user", "follow up"),
  };
  auto prior = extractPriorTurnPrefix(withTool);
  ASSERT_TRUE(prior.has_value());
  ASSERT_EQ(prior->size(), 1u);
  EXPECT_EQ((*prior)[0].role, "user");
  EXPECT_EQ((*prior)[0].content, "q");
}

TEST(ConversationHasherLogic, HashConversationPrefix_EmptyIsZero) {
  EXPECT_EQ(hashConversationPrefix({}), 0u);
}

}  // namespace

// ---------------------------------------------------------------------------
// Tokenizer-backed tests
// ---------------------------------------------------------------------------

class ConversationHasherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string path = tt::config::tokenizerPath();
    if (path.empty()) {
      GTEST_SKIP()
          << "Tokenizer path not configured (set model / tokenizer path)";
    }
    // exercise the same model path the server uses; activeTokenizer is static
    const auto& tok = activeTokenizer();
    if (!tok.isLoaded()) {
      GTEST_SKIP() << "Active tokenizer not loaded for path: " << path;
    }
  }
};

TEST_F(ConversationHasherTest, HashConversationPrefix_IsDeterministic) {
  std::vector<ChatMessage> prefix = {
      makeMessage("user", "hello hasher"),
  };
  uint64_t a = hashConversationPrefix(prefix);
  uint64_t b = hashConversationPrefix(prefix);
  EXPECT_EQ(a, b);
}

TEST_F(ConversationHasherTest, HashConversationPrefix_DiffersForContent) {
  uint64_t h1 = hashConversationPrefix({makeMessage("user", "A")});
  uint64_t h2 = hashConversationPrefix({makeMessage("user", "B")});
  EXPECT_NE(h1, h2);
}

TEST_F(ConversationHasherTest, RenderLastUserTurn_PicksLastUser) {
  std::vector<ChatMessage> thread = {
      makeMessage("user", "older"),
      makeMessage("assistant", "reply"),
      makeMessage("user", "newer"),
  };
  std::string delta = renderLastUserTurn(thread);
  EXPECT_FALSE(delta.empty());
  // Last user content should appear in the rendered single-turn template
  EXPECT_NE(delta.find("newer"), std::string::npos);
  EXPECT_EQ(delta.find("older"), std::string::npos);
}

TEST_F(ConversationHasherTest, RenderLastUserTurn_NoUserRoleReturnsEmpty) {
  EXPECT_EQ(renderLastUserTurn({makeMessage("assistant", "no user")}), "");
}

TEST_F(ConversationHasherTest,
       ComputePrefixCachingInfo_AlignedWithDecomposedPipeline) {
  // Primary regression target: computePrefixCachingInfo() must stay consistent
  // with strip / hash / render / prior extraction.
  std::vector<ChatMessage> messages = {
      makeMessage("system", "You are a test bot"),
      makeMessage("user", "first turn"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second turn"),
  };
  std::vector<ChatMessage> turns = stripToolMessages(messages);

  PrefixCachingInfo info = computePrefixCachingInfo(messages);

  EXPECT_EQ(info.deltaPrompt, renderLastUserTurn(turns));
  EXPECT_EQ(info.registrationHash, hashConversationPrefix(turns));

  std::optional<std::vector<ChatMessage>> prior =
      extractPriorTurnPrefix(messages);
  if (prior.has_value()) {
    EXPECT_TRUE(info.hasPriorTurn);
    ASSERT_TRUE(info.lookupHash.has_value());
    EXPECT_EQ(*info.lookupHash, hashConversationPrefix(*prior));
  } else {
    EXPECT_FALSE(info.hasPriorTurn);
    EXPECT_FALSE(info.lookupHash.has_value());
  }
}

TEST_F(ConversationHasherTest, ComputePrefixCachingInfo_SingleUserNoPrior) {
  std::vector<ChatMessage> messages = {makeMessage("user", "solo")};
  PrefixCachingInfo info = computePrefixCachingInfo(messages);

  EXPECT_FALSE(info.hasPriorTurn);
  EXPECT_FALSE(info.lookupHash.has_value());
  EXPECT_EQ(info.registrationHash,
            hashConversationPrefix(stripToolMessages(messages)));
  EXPECT_EQ(info.deltaPrompt, renderLastUserTurn(stripToolMessages(messages)));
}

TEST_F(ConversationHasherTest, ComputePrefixCachingInfo_MultiTurnHasLookup) {
  std::vector<ChatMessage> messages = {
      makeMessage("user", "q1"),
      makeMessage("assistant", "a1"),
      makeMessage("user", "q2"),
  };
  PrefixCachingInfo info = computePrefixCachingInfo(messages);
  std::vector<ChatMessage> turns = stripToolMessages(messages);

  EXPECT_TRUE(info.hasPriorTurn);
  ASSERT_TRUE(info.lookupHash.has_value());
  std::vector<ChatMessage> prior = {makeMessage("user", "q1")};
  EXPECT_EQ(*info.lookupHash, hashConversationPrefix(prior));
  EXPECT_NE(info.registrationHash, *info.lookupHash);
  EXPECT_EQ(info.registrationHash, hashConversationPrefix(turns));
}

TEST_F(ConversationHasherTest,
       ComputePrefixCachingInfo_AssistantUserOnlyHasNoLookup) {
  std::vector<ChatMessage> messages = {
      makeMessage("assistant", "hi"),
      makeMessage("user", "u"),
  };
  PrefixCachingInfo info = computePrefixCachingInfo(messages);

  EXPECT_FALSE(info.hasPriorTurn);
  EXPECT_FALSE(info.lookupHash.has_value());
}

TEST_F(ConversationHasherTest, ComputePrefixCachingInfo_StripsTools) {
  std::vector<ChatMessage> messages = {
      makeMessage("user", "q1"),
      makeMessage("assistant", "call tool"),
      makeMessage("tool", "json result"),
      makeMessage("user", "q2"),
  };
  std::vector<ChatMessage> noTool = {
      makeMessage("user", "q1"),
      makeMessage("assistant", "call tool"),
      makeMessage("user", "q2"),
  };

  PrefixCachingInfo got = computePrefixCachingInfo(messages);
  PrefixCachingInfo noToolInfo = computePrefixCachingInfo(noTool);

  // Stripped view matches a conversation without the tool line.
  EXPECT_EQ(got.registrationHash, noToolInfo.registrationHash);
  EXPECT_EQ(got.deltaPrompt, noToolInfo.deltaPrompt);
  EXPECT_EQ(got.hasPriorTurn, noToolInfo.hasPriorTurn);
  if (got.lookupHash.has_value()) {
    ASSERT_TRUE(noToolInfo.lookupHash.has_value());
    EXPECT_EQ(*got.lookupHash, *noToolInfo.lookupHash);
  } else {
    EXPECT_FALSE(noToolInfo.lookupHash.has_value());
  }
}

TEST_F(ConversationHasherTest, ComputePrefixCachingInfo_StabilitySecondTurn) {
  // After turn 1, "registration" is the one-message conversation; add assistant
  // + user and ensure lookup would find the first registration hash.
  std::vector<ChatMessage> turn1 = {makeMessage("user", "first")};
  uint64_t hTurn1 = hashConversationPrefix(turn1);

  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "first"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second"),
  };
  PrefixCachingInfo info2 = computePrefixCachingInfo(turn2);
  ASSERT_TRUE(info2.hasPriorTurn);
  ASSERT_TRUE(info2.lookupHash.has_value());
  EXPECT_EQ(*info2.lookupHash, hTurn1);
  EXPECT_EQ(info2.deltaPrompt, renderLastUserTurn(stripToolMessages(turn2)));
}
