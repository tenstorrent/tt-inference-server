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

// ---------------------------------------------------------------------------
// Think-row accounting (getPrefixCacheHashesByBlocksWithThinking)
//
// accumulatedThinkTokens is the correction that turns a block-aligned match
// back into a KV row index: matched_tokens + accumulatedThinkTokens ==
// kv_position_id, the first free KV index. So it must count every row the NEXT
// turn's prompt will not re-supply — the reasoning content always, and each
// delimiter only when this model's chat template drops it from history.
//
// The hashes must be identical under every policy: matching has to work across
// a prompt that no longer carries the reasoning at all.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kThinkStart = 90001;
constexpr uint32_t kThinkEnd = 90002;

// [first block of ordinary tokens] <think> t t </think> [one more block]
std::vector<uint32_t> tokensWithOneThinkBlock() {
  const size_t firstBlockSize = tt::config::prefixCacheFirstBlockSize();
  const size_t blockSize = tt::config::prefixCacheBlockSize();
  std::vector<uint32_t> tokens;
  tokens.reserve(firstBlockSize + blockSize + 4);
  for (size_t i = 0; i < firstBlockSize; ++i) {
    tokens.push_back(static_cast<uint32_t>(1000 + i));
  }
  tokens.push_back(kThinkStart);
  tokens.push_back(70001);  // reasoning content
  tokens.push_back(70002);  // reasoning content
  tokens.push_back(kThinkEnd);
  for (size_t i = 0; i < blockSize; ++i) {
    tokens.push_back(static_cast<uint32_t>(2000 + i));
  }
  return tokens;
}

std::vector<BlockHashInfo> hashWithPolicy(bool startInHistory,
                                          bool endInHistory) {
  return getPrefixCacheHashesByBlocksWithThinking(tokensWithOneThinkBlock(),
                                                  kThinkStart, kThinkEnd,
                                                  startInHistory, endInHistory);
}

}  // namespace

TEST(ConversationHasherThinkRows, CountsDelimitersDroppedFromHistory) {
  // DeepSeek / Gemma / MiniMax-M2.7 shape: the next prompt carries neither
  // delimiter, so both rows are this counter's responsibility.
  auto blocks =
      hashWithPolicy(/*startInHistory=*/false, /*endInHistory=*/false);
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_EQ(blocks[0].accumulatedThinkTokens, 0u)
      << "the think block starts after the first block closes";
  EXPECT_EQ(blocks[1].accumulatedThinkTokens, 4u)
      << "2 reasoning tokens + both delimiters occupy KV rows the next "
         "prompt will not contain";
}

TEST(ConversationHasherThinkRows, SkipsDelimitersKeptInHistory) {
  // Kimi / GLM-5.2 shape: the template re-renders `<think></think>`, so those
  // two rows arrive with the next prompt and must NOT be counted again.
  auto blocks = hashWithPolicy(/*startInHistory=*/true, /*endInHistory=*/true);
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_EQ(blocks[1].accumulatedThinkTokens, 2u)
      << "only the reasoning content is missing from the next prompt";
}

TEST(ConversationHasherThinkRows, CountsEachDelimiterIndependently) {
  // GLM-5.1 / MiniMax-M3 shape: history keeps the closing delimiter only.
  auto blocks = hashWithPolicy(/*startInHistory=*/false, /*endInHistory=*/true);
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_EQ(blocks[1].accumulatedThinkTokens, 3u)
      << "2 reasoning tokens + the dropped opening delimiter";
}

TEST(ConversationHasherThinkRows, PolicyNeverChangesTheHashes) {
  // The fingerprint side must stay policy-independent, otherwise a prompt that
  // renders the delimiters differently stops matching its own session.
  auto dropped = hashWithPolicy(false, false);
  auto kept = hashWithPolicy(true, true);
  auto mixed = hashWithPolicy(false, true);
  ASSERT_EQ(dropped.size(), kept.size());
  ASSERT_EQ(dropped.size(), mixed.size());
  for (size_t i = 0; i < dropped.size(); ++i) {
    EXPECT_EQ(dropped[i].hash, kept[i].hash) << "block " << i;
    EXPECT_EQ(dropped[i].hash, mixed[i].hash) << "block " << i;
  }
}

TEST(ConversationHasherThinkRows, MatchedPlusThinkEqualsKvRows) {
  // The invariant the accounting exists to uphold, stated directly: for a
  // prompt that re-renders neither delimiter, the matched (hashed) token count
  // plus the think rows must equal the number of KV rows consumed.
  const size_t firstBlockSize = tt::config::prefixCacheFirstBlockSize();
  const size_t blockSize = tt::config::prefixCacheBlockSize();
  const auto tokens = tokensWithOneThinkBlock();

  auto blocks =
      hashWithPolicy(/*startInHistory=*/false, /*endInHistory=*/false);
  ASSERT_EQ(blocks.size(), 2u);

  const size_t matchedTokens = firstBlockSize + blockSize;  // hashed tokens
  EXPECT_EQ(matchedTokens + blocks.back().accumulatedThinkTokens, tokens.size())
      << "kv_position_id must land on the first free KV row";
}
