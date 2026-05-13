// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/resolvers/chat_completions_resolver.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "api/resolvers/resolved_session.hpp"
#include "config/settings.hpp"
#include "domain/llm/chat_message.hpp"
#include "domain/session.hpp"
#include "services/session_manager.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace {

using tt::api::resolvers::ChatCompletionsResolver;
using tt::api::resolvers::ResolvedSession;
using tt::api::resolvers::SessionError;
using tt::api::resolvers::SessionErrorType;
using tt::domain::llm::ChatMessage;
using tt::utils::tokenizers::activeTokenizer;

ChatMessage makeMessage(std::string role, std::string content) {
  ChatMessage m;
  m.role = std::move(role);
  m.content = std::move(content);
  return m;
}

// Trantor requires an EventLoop to be both created and run on the same thread.
struct LoopFixture {
  std::promise<trantor::EventLoop*> promise_;
  trantor::EventLoop* loop{nullptr};
  std::thread loopThread;

  LoopFixture() {
    auto future = promise_.get_future();
    loopThread = std::thread([this]() {
      trantor::EventLoop eventLoop;
      promise_.set_value(&eventLoop);
      eventLoop.loop();
    });
    loop = future.get();
  }

  ~LoopFixture() {
    if (loop) loop->quit();
    if (loopThread.joinable()) loopThread.join();
  }
};

// Pre-assigned slot bypasses IPC and returns a session synchronously
// (queued on the caller's loop). This is the same pattern used by
// session_manager_test.
std::string createSessionWithSlot(tt::services::SessionManager& manager,
                                  trantor::EventLoop* loop, uint32_t slotId) {
  std::promise<std::string> promise;
  auto future = promise.get_future();

  manager.createSession(
      [&promise](const tt::domain::Session& s) {
        promise.set_value(s.getSessionId());
      },
      [&promise](std::string_view err) {
        promise.set_exception(
            std::make_exception_ptr(std::runtime_error(std::string(err))));
      },
      loop, 0, slotId);

  return future.get();
}

// ---------------------------------------------------------------------------
// Resolver behavior when no SessionManager is wired up. Exercised on every
// build because it short-circuits before reaching the tokenizer.
// ---------------------------------------------------------------------------

TEST(ChatCompletionsResolverNoManager, ReturnsEmptyResolvedSession) {
  ChatCompletionsResolver resolver(/*sessionManager=*/nullptr);

  bool calledOnDone = false;
  ResolvedSession got;
  resolver.resolve(
      {makeMessage("user", "hello")}, /*loop=*/nullptr,
      [&](ResolvedSession r) {
        calledOnDone = true;
        got = std::move(r);
      },
      [](const SessionError&) { FAIL() << "expected onDone, got onError"; },
      /*cancelFn=*/nullptr);

  EXPECT_TRUE(calledOnDone);
  EXPECT_FALSE(got.sessionId.has_value());
  EXPECT_FALSE(got.slotId.has_value());
  EXPECT_EQ(got.session, nullptr);
  EXPECT_TRUE(got.isFresh);
  EXPECT_EQ(got.registrationHash, 0u);
  EXPECT_TRUE(got.prompt.empty());
  EXPECT_EQ(got.promptTokensCount, 0);
}

// ---------------------------------------------------------------------------
// Tokenizer-backed resolver tests. These need a real chat template to
// produce stable hashes. The fresh-allocation path (no slotId hint) goes
// through IPC and is exercised by main_integration_test instead.
// ---------------------------------------------------------------------------

class ChatCompletionsResolverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (tt::config::tokenizerPath().empty()) {
      GTEST_SKIP() << "Tokenizer path not configured";
    }
    if (!activeTokenizer().isLoaded()) {
      GTEST_SKIP() << "Active tokenizer not loaded";
    }
    sessionManager = std::make_shared<tt::services::SessionManager>();
    resolver = std::make_unique<ChatCompletionsResolver>(sessionManager);
  }

  std::shared_ptr<tt::services::SessionManager> sessionManager;
  std::unique_ptr<ChatCompletionsResolver> resolver;
};

TEST_F(ChatCompletionsResolverTest, PrefixHashHit_ReturnsExistingSession) {
  LoopFixture lf;

  // Stand up a session under the hash that turn 1 would register.
  auto sessionId = createSessionWithSlot(*sessionManager, lf.loop, 17u);
  ASSERT_FALSE(sessionId.empty());

  std::vector<ChatMessage> turn1 = {makeMessage("user", "first")};
  const uint64_t hTurn1 = tt::utils::hashConversationPrefix(turn1);
  sessionManager->registerPrefixHash(sessionId, hTurn1);

  // Resolve a turn-2 conversation; its lookupHash should match the
  // registered turn-1 hash.
  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "first"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second"),
  };

  bool calledOnDone = false;
  ResolvedSession got;
  resolver->resolve(
      turn2, lf.loop,
      [&](ResolvedSession r) {
        calledOnDone = true;
        got = std::move(r);
      },
      [](const SessionError& e) {
        FAIL() << "expected onDone, got onError: " << e.message;
      },
      /*cancelFn=*/nullptr);

  EXPECT_TRUE(calledOnDone);
  ASSERT_TRUE(got.sessionId.has_value());
  EXPECT_EQ(*got.sessionId, sessionId);
  ASSERT_TRUE(got.slotId.has_value());
  EXPECT_EQ(*got.slotId, 17u);
  ASSERT_NE(got.session, nullptr);
  EXPECT_FALSE(got.isFresh);
  EXPECT_EQ(got.registrationHash, tt::utils::hashConversationPrefix(turn2));
  // delta prompt is the rendered last user turn.
  EXPECT_FALSE(got.prompt.empty());
  EXPECT_NE(got.prompt.find("second"), std::string::npos);
  EXPECT_GT(got.promptTokensCount, 0);

  // Release in-flight so the LoopFixture teardown can close the session.
  got.session->clearInFlight();
}

TEST_F(ChatCompletionsResolverTest, AllSessionsInFlight_EmitsRateLimit) {
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(*sessionManager, lf.loop, 18u);
  ASSERT_FALSE(sessionId.empty());

  std::vector<ChatMessage> turn1 = {makeMessage("user", "first")};
  const uint64_t hTurn1 = tt::utils::hashConversationPrefix(turn1);
  sessionManager->registerPrefixHash(sessionId, hTurn1);

  // Mark the only candidate session in-flight before resolving.
  sessionManager->acquireInFlight(sessionId, /*cancelFn=*/nullptr);

  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "first"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second"),
  };

  bool calledOnError = false;
  SessionError err;
  resolver->resolve(
      turn2, lf.loop,
      [](ResolvedSession) { FAIL() << "expected RATE_LIMIT error"; },
      [&](const SessionError& e) {
        calledOnError = true;
        err = e;
      },
      /*cancelFn=*/nullptr);

  EXPECT_TRUE(calledOnError);
  EXPECT_EQ(err.type, SessionErrorType::RATE_LIMIT);

  auto* session = sessionManager->getSession(sessionId);
  ASSERT_NE(session, nullptr);
  session->clearInFlight();
}

TEST_F(ChatCompletionsResolverTest, PrefixHashHit_PromotesNextTurnHash) {
  // After a HIT the resolver re-registers the session under the full
  // current conversation's hash so the *next* turn can find it.
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(*sessionManager, lf.loop, 19u);
  ASSERT_FALSE(sessionId.empty());

  std::vector<ChatMessage> turn1 = {makeMessage("user", "first")};
  const uint64_t hTurn1 = tt::utils::hashConversationPrefix(turn1);
  sessionManager->registerPrefixHash(sessionId, hTurn1);

  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "first"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second"),
  };
  const uint64_t hTurn2 = tt::utils::hashConversationPrefix(turn2);

  resolver->resolve(
      turn2, lf.loop,
      [](ResolvedSession r) {
        if (r.session) r.session->clearInFlight();
      },
      [](const SessionError& e) { FAIL() << e.message; },
      /*cancelFn=*/nullptr);

  // Look up by the turn-2 hash should now return the same session (after
  // marking it idle above) — verify by trying to acquire it.
  auto acquired =
      sessionManager->tryAcquireByPrefixHash(hTurn2, /*cancelFn=*/nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->sessionId, sessionId);

  auto* session = sessionManager->getSession(sessionId);
  ASSERT_NE(session, nullptr);
  session->clearInFlight();
}

}  // namespace
