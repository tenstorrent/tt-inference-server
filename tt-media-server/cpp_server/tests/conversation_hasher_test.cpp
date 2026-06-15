// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/conversation_hasher.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "config/settings.hpp"
#include "domain/llm/chat_message.hpp"
#include "utils/tokenizers/tokenizer.hpp"

using namespace tt::domain;
using namespace tt::domain::llm;
using namespace tt::utils;
using tt::utils::tokenizers::activeTokenizer;

namespace {

using namespace tt::domain::llm;

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
  std::string delta = renderLastUserTurn(thread, /*hasPriorTurn=*/true);
  EXPECT_FALSE(delta.empty());
  // Last user content should appear in the rendered single-turn template
  EXPECT_NE(delta.find("newer"), std::string::npos);
  EXPECT_EQ(delta.find("older"), std::string::npos);
}

TEST_F(ConversationHasherTest, RenderLastUserTurn_NoUserRoleReturnsEmpty) {
  EXPECT_EQ(renderLastUserTurn({makeMessage("assistant", "no user")},
                               /*hasPriorTurn=*/false),
            "");
}

TEST_F(ConversationHasherTest, RenderLastUserTurn_BosIncludedOnlyWithoutPrior) {
  auto cfg = tt::utils::tokenizers::getTokenizerConfig();
  if (!cfg.add_bos_token || cfg.bos_token.empty()) {
    GTEST_SKIP() << "Tokenizer config does not add a BOS token";
  }

  std::vector<ChatMessage> lastUser = {makeMessage("user", "first turn")};
  std::string freshDelta = renderLastUserTurn(lastUser, /*hasPriorTurn=*/false);
  EXPECT_EQ(freshDelta.compare(0, cfg.bos_token.size(), cfg.bos_token), 0)
      << "Fresh sessions should keep BOS at the start of the delta";

  std::string contDelta = renderLastUserTurn(lastUser, /*hasPriorTurn=*/true);
  EXPECT_NE(contDelta.compare(0, cfg.bos_token.size(), cfg.bos_token), 0)
      << "Continuations must not duplicate BOS already in the KV cache";
}
