// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/resolvers/chat_completions_resolver.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "api/resolvers/session_error.hpp"
#include "config/settings.hpp"
#include "domain/llm/chat_message.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/session.hpp"
#include "services/session_manager.hpp"
#include "services/slot_lease.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace {

using tt::api::resolvers::ChatCompletionsResolver;
using tt::api::resolvers::SessionError;
using tt::api::resolvers::SessionErrorType;
using tt::domain::llm::ChatMessage;
using tt::domain::llm::LLMRequest;
using tt::services::SlotLease;
using tt::utils::tokenizers::activeTokenizer;

ChatMessage makeMessage(std::string role, std::string content) {
  ChatMessage m;
  m.role = std::move(role);
  m.content = std::move(content);
  return m;
}

std::shared_ptr<LLMRequest> makeRequest(std::vector<ChatMessage> messages) {
  auto req = std::make_shared<LLMRequest>(/*taskId=*/0u);
  req->messages = std::move(messages);
  req->prompt = std::string{};  // resolver only writes prompt on a HIT
  return req;
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
// hashMessages: structural identity, no tokenizer dependency.
// ---------------------------------------------------------------------------

TEST(ChatCompletionsResolverHash, EmptyIsZero) {
  EXPECT_EQ(ChatCompletionsResolver::hashMessages({}), 0u);
}

TEST(ChatCompletionsResolverHash, DeterministicAndContentSensitive) {
  std::vector<ChatMessage> a = {makeMessage("user", "hello")};
  std::vector<ChatMessage> b = {makeMessage("user", "hello")};
  std::vector<ChatMessage> c = {makeMessage("user", "Hello")};
  std::vector<ChatMessage> d = {makeMessage("assistant", "hello")};

  EXPECT_EQ(ChatCompletionsResolver::hashMessages(a),
            ChatCompletionsResolver::hashMessages(b));
  EXPECT_NE(ChatCompletionsResolver::hashMessages(a),
            ChatCompletionsResolver::hashMessages(c));
  EXPECT_NE(ChatCompletionsResolver::hashMessages(a),
            ChatCompletionsResolver::hashMessages(d));
}

TEST(ChatCompletionsResolverHash, BoundaryNotAmbiguous) {
  // (role="us", content="er.x") must not collide with (role="user",
  // content=".x")
  std::vector<ChatMessage> a = {makeMessage("us", "er.x")};
  std::vector<ChatMessage> b = {makeMessage("user", ".x")};
  EXPECT_NE(ChatCompletionsResolver::hashMessages(a),
            ChatCompletionsResolver::hashMessages(b));
}

// ---------------------------------------------------------------------------
// Resolver behavior when no SessionManager is wired up. Exercised on every
// build because it short-circuits before reaching the tokenizer.
// ---------------------------------------------------------------------------

TEST(ChatCompletionsResolverNoManager, LeavesRequestUntouched) {
  ChatCompletionsResolver resolver(/*manager=*/nullptr);

  auto req = makeRequest({makeMessage("user", "hello")});
  req->prompt = std::string{"original prompt"};
  req->continuation = true;  // sentinel: resolver must not flip this
  req->registrationHash = 42u;

  bool calledOnDone = false;
  bool leaseWasEmpty = false;
  resolver.resolve(
      req, /*loop=*/nullptr,
      [&](SlotLease lease) {
        calledOnDone = true;
        leaseWasEmpty = lease.empty();
      },
      [](const SessionError&) { FAIL() << "expected onDone, got onError"; },
      /*cancelFn=*/nullptr);

  EXPECT_TRUE(calledOnDone);
  EXPECT_TRUE(leaseWasEmpty);

  // Request was not mutated.
  EXPECT_FALSE(req->sessionId.has_value());
  EXPECT_FALSE(req->slotId.has_value());
  EXPECT_TRUE(req->continuation);
  EXPECT_EQ(req->registrationHash, 42u);
  ASSERT_TRUE(std::holds_alternative<std::string>(req->prompt));
  EXPECT_EQ(std::get<std::string>(req->prompt), "original prompt");
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

TEST_F(ChatCompletionsResolverTest, PrefixHashHit_MutatesRequestWithDelta) {
  LoopFixture lf;

  // Stand up a session under the hash that turn 1 would register.
  auto sessionId = createSessionWithSlot(*sessionManager, lf.loop, 17u);
  ASSERT_FALSE(sessionId.empty());

  std::vector<ChatMessage> turn1 = {makeMessage("user", "first")};
  const uint64_t hTurn1 = ChatCompletionsResolver::hashMessages(turn1);
  sessionManager->registerPrefixHash(sessionId, hTurn1);

  // Turn 2's lookupHash matches the registered turn-1 hash.
  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "first"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second"),
  };
  auto req = makeRequest(turn2);
  req->prompt = std::string{"full-conversation prompt"};

  bool calledOnDone = false;
  SlotLease handedOut;
  resolver->resolve(
      req, lf.loop,
      [&](SlotLease lease) {
        calledOnDone = true;
        handedOut = std::move(lease);
      },
      [](const SessionError& e) {
        FAIL() << "expected onDone, got onError: " << e.message;
      },
      /*cancelFn=*/nullptr);

  EXPECT_TRUE(calledOnDone);

  // Request must carry the cached session / slot and the delta prompt.
  ASSERT_TRUE(req->sessionId.has_value());
  EXPECT_EQ(*req->sessionId, sessionId);
  ASSERT_TRUE(req->slotId.has_value());
  EXPECT_EQ(*req->slotId, 17u);
  EXPECT_TRUE(req->continuation);
  EXPECT_EQ(req->registrationHash,
            ChatCompletionsResolver::hashMessages(turn2));
  ASSERT_TRUE(std::holds_alternative<std::string>(req->prompt));
  EXPECT_NE(std::get<std::string>(req->prompt), "full-conversation prompt");
  EXPECT_NE(std::get<std::string>(req->prompt).find("second"),
            std::string::npos);
  EXPECT_GT(req->prompt_tokens_count, 0);

  // Lease binds the right session/slot and carries the in-flight grant.
  ASSERT_FALSE(handedOut.empty());
  EXPECT_EQ(handedOut.sessionId(), sessionId);
  EXPECT_EQ(handedOut.slotId(), 17u);
  // Dropping the lease releases the session so teardown can close it.
}

TEST_F(ChatCompletionsResolverTest, AllSessionsInFlight_EmitsRateLimit) {
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(*sessionManager, lf.loop, 18u);
  ASSERT_FALSE(sessionId.empty());

  std::vector<ChatMessage> turn1 = {makeMessage("user", "first")};
  const uint64_t hTurn1 = ChatCompletionsResolver::hashMessages(turn1);
  sessionManager->registerPrefixHash(sessionId, hTurn1);

  // Mark the only candidate session in-flight before resolving.
  sessionManager->acquireInFlight(sessionId, /*cancelFn=*/nullptr);

  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "first"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second"),
  };
  auto req = makeRequest(turn2);
  req->prompt = std::string{"full-conversation prompt"};

  bool calledOnError = false;
  SessionError err;
  resolver->resolve(
      req, lf.loop,
      [](SlotLease) { FAIL() << "expected RATE_LIMIT error"; },
      [&](const SessionError& e) {
        calledOnError = true;
        err = e;
      },
      /*cancelFn=*/nullptr);

  EXPECT_TRUE(calledOnError);
  EXPECT_EQ(err.type, SessionErrorType::RATE_LIMIT);

  // Request must remain in its pre-resolve state — no session/slot bound.
  EXPECT_FALSE(req->sessionId.has_value());
  EXPECT_FALSE(req->slotId.has_value());
  EXPECT_FALSE(req->continuation);
  EXPECT_EQ(req->registrationHash, 0u);

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
  const uint64_t hTurn1 = ChatCompletionsResolver::hashMessages(turn1);
  sessionManager->registerPrefixHash(sessionId, hTurn1);

  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "first"),
      makeMessage("assistant", "ack"),
      makeMessage("user", "second"),
  };
  const uint64_t hTurn2 = ChatCompletionsResolver::hashMessages(turn2);
  auto req = makeRequest(turn2);

  resolver->resolve(
      req, lf.loop,
      [](SlotLease /*lease*/) {
        // Lease falls out of scope here, releasing the in-flight grant.
      },
      [](const SessionError& e) { FAIL() << e.message; },
      /*cancelFn=*/nullptr);

  // Looking up by the turn-2 hash should now return the same session.
  auto acquired =
      sessionManager->tryAcquireByPrefixHash(hTurn2, /*cancelFn=*/nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->sessionId, sessionId);

  auto* session = sessionManager->getSession(sessionId);
  ASSERT_NE(session, nullptr);
  session->clearInFlight();
}

TEST_F(ChatCompletionsResolverTest, PrefixHashHit_IgnoresToolMessages) {
  // Tool turns sitting between assistant + user must NOT change prefix
  // identity: a session registered under [user "q1"] should be found by
  // a turn-2 conversation that includes a tool turn before the new user.
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(*sessionManager, lf.loop, 20u);
  ASSERT_FALSE(sessionId.empty());

  std::vector<ChatMessage> priorPrefix = {makeMessage("user", "q1")};
  sessionManager->registerPrefixHash(
      sessionId, ChatCompletionsResolver::hashMessages(priorPrefix));

  ChatMessage toolTurn;
  toolTurn.role = "tool";
  toolTurn.content = "tool result blob";
  toolTurn.tool_call_id = "call_abc";

  std::vector<ChatMessage> turn2 = {
      makeMessage("user", "q1"),
      makeMessage("assistant", "let me call a tool"),
      toolTurn,
      makeMessage("user", "follow up"),
  };

  auto req = makeRequest(turn2);
  resolver->resolve(
      req, lf.loop, [](SlotLease /*lease*/) {},
      [](const SessionError& e) { FAIL() << e.message; },
      /*cancelFn=*/nullptr);

  ASSERT_TRUE(req->sessionId.has_value());
  EXPECT_EQ(*req->sessionId, sessionId);
  EXPECT_TRUE(req->continuation);
}

}  // namespace
