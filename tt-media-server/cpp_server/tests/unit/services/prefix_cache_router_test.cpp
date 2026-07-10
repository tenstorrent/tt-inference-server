// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/prefix_cache_router.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "domain/session.hpp"

namespace {

struct TestableSession : tt::domain::Session {
  using tt::domain::Session::clearInFlight;
  using tt::domain::Session::markInFlight;
  using tt::domain::Session::markPrepared;

  explicit TestableSession(uint32_t slotId, size_t hash = 0)
      : tt::domain::Session(slotId, hash) {}
};

class PrefixCacheRouterTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Must be set before logger init caches llmMode() on first use.
    setenv("LLM_MODE", "decode", 1);
  }

  void SetUp() override {
    setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);
    setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
    setenv("KV_CACHE_BLOCK_SIZE", "32", 1);

    sessions_.clear();
    inFlight_.clear();
    lockedSlots_.clear();
    shrunkSessions_.clear();
    createdSlotCopyFrom_.clear();
    nextAllocSlot_ = 1000;

    router =
        std::make_unique<tt::services::PrefixCacheRouter>(makeCallbacks());
  }

  std::string addSession(uint32_t slotId, uint64_t hash = 0) {
    auto session = std::make_shared<TestableSession>(slotId, hash);
    session->markPrepared();
    const auto sessionId = session->getSessionId();
    sessions_.emplace(sessionId, std::move(session));
    return sessionId;
  }

  void registerBlocks(const std::string& sessionId,
                      const std::vector<tt::utils::BlockHashInfo>& blocks) {
    router->registerPrefixHash(sessionId, blocks);
  }

  void releaseSession(const std::string& sessionId) {
    auto it = sessions_.find(sessionId);
    if (it != sessions_.end()) {
      it->second->clearInFlight();
      inFlight_.erase(sessionId);
    }
  }

  std::vector<tt::utils::BlockHashInfo> threeBlocks() const {
    return {
        {100, 0},
        {200, 0},
        {300, 0},
    };
  }

  std::vector<tt::utils::BlockHashInfo> fourBlocks() const {
    return {
        {100, 0},
        {200, 0},
        {300, 0},
        {400, 0},
    };
  }

  std::vector<uint32_t> makeThreeBlockPrompt() const {
    std::vector<uint32_t> prompt(32 + 32 + 32);
    std::iota(prompt.begin(), prompt.end(), 0);
    return prompt;
  }

  std::vector<uint32_t> makeFourBlockPrompt() const {
    auto prompt = makeThreeBlockPrompt();
    std::vector<uint32_t> tail(32);
    std::iota(tail.begin(), tail.end(), static_cast<uint32_t>(prompt.size()));
    prompt.insert(prompt.end(), tail.begin(), tail.end());
    return prompt;
  }

  tt::domain::MarkInFlightResult tryMarkSessionInFlight(
      const std::string& sessionId, std::function<void()>& cancelFn,
      std::optional<uint64_t> expectedKeyHash,
      const std::string* expectedResponseId) {
    tt::domain::MarkInFlightResult result;
    auto it = sessions_.find(sessionId);
    if (it == sessions_.end()) {
      result.outcome = tt::domain::MarkInFlightOutcome::NotFound;
      return result;
    }
    auto& session = *it->second;
    if (expectedKeyHash.has_value() && session.getHash() != *expectedKeyHash) {
      result.outcome = tt::domain::MarkInFlightOutcome::Stale;
      return result;
    }
    if (expectedResponseId != nullptr &&
        session.getResponseId() != *expectedResponseId) {
      result.outcome = tt::domain::MarkInFlightOutcome::Stale;
      return result;
    }
    if (session.isInFlight()) {
      result.outcome = tt::domain::MarkInFlightOutcome::Busy;
      result.slotId = session.getSlotId();
      return result;
    }
    session.markInFlight();
    session.setCancelFn(std::move(cancelFn));
    inFlight_.insert(sessionId);
    result.outcome = tt::domain::MarkInFlightOutcome::Marked;
    result.slotId = session.getSlotId();
    return result;
  }

  tt::services::PrefixCacheRouterCallbacks makeCallbacks() {
    tt::services::PrefixCacheRouterCallbacks callbacks;

    callbacks.tryMarkInFlight = [this](const std::string& sessionId,
                                       std::function<void()>& cancelFn,
                                       std::optional<uint64_t> expectedKeyHash,
                                       const std::string* expectedResponseId)
        -> tt::domain::MarkInFlightResult {
      return tryMarkSessionInFlight(sessionId, cancelFn, expectedKeyHash,
                                    expectedResponseId);
    };

    callbacks.getSession = [this](const std::string& sessionId) {
      auto it = sessions_.find(sessionId);
      return it == sessions_.end() ? nullptr : it->second;
    };

    callbacks.getSessionHash = [this](const std::string& sessionId) {
      auto it = sessions_.find(sessionId);
      if (it == sessions_.end()) {
        return std::optional<uint64_t>{};
      }
      return std::optional<uint64_t>{it->second->getHash()};
    };

    callbacks.setSessionHash = [this](const std::string& sessionId,
                                      uint64_t keyHash) {
      auto it = sessions_.find(sessionId);
      if (it == sessions_.end()) {
        return false;
      }
      it->second->setHash(keyHash);
      return true;
    };

    callbacks.setSessionResponseId = [this](const std::string& sessionId,
                                            const std::string& responseId) {
      auto it = sessions_.find(sessionId);
      if (it == sessions_.end()) {
        return false;
      }
      it->second->setResponseId(responseId);
      return true;
    };

    callbacks.onSessionInFlight = [] {
      throw std::runtime_error("session in flight");
    };

    callbacks.createSession =
        [this](std::function<void(const tt::domain::Session&)> onCompletion,
               std::function<void(std::string_view)> onError,
               trantor::EventLoop* /*eventLoop*/,
               std::vector<tt::utils::BlockHashInfo> initialBlockInfos,
               std::optional<uint32_t> slotIdToCopyFrom) {
          if (slotIdToCopyFrom.has_value()) {
            createdSlotCopyFrom_.push_back(*slotIdToCopyFrom);
          }
          auto session = std::make_shared<TestableSession>(
              nextAllocSlot_++,
              initialBlockInfos.empty() ? 0 : initialBlockInfos.front().hash);
          session->markPrepared();
          sessions_.emplace(session->getSessionId(), session);
          onCompletion(*session);
        };

    callbacks.acquireInFlight = [this](const std::string& sessionId,
                                       std::function<void()> cancelFn) {
      return tryMarkSessionInFlight(sessionId, cancelFn, std::nullopt, nullptr)
          .slotId;
    };

    callbacks.lockSlot = [this](uint32_t slotId) {
      lockedSlots_.insert(slotId);
    };
    callbacks.unlockSlot = [this](uint32_t slotId) {
      lockedSlots_.erase(slotId);
    };

    callbacks.shrinkResidentPrefixToMatchedTokens =
        [this](const std::string& sessionId, uint32_t matchedTokens) {
          shrunkSessions_.emplace_back(sessionId, matchedTokens);
        };

    return callbacks;
  }

  std::unordered_map<std::string, std::shared_ptr<TestableSession>> sessions_;
  std::unordered_set<std::string> inFlight_;
  std::unordered_set<uint32_t> lockedSlots_;
  std::vector<std::pair<std::string, uint32_t>> shrunkSessions_;
  std::vector<uint32_t> createdSlotCopyFrom_;
  uint32_t nextAllocSlot_ = 1000;
  std::unique_ptr<tt::services::PrefixCacheRouter> router;
};

// ---------------------------------------------------------------------------
// tryAcquireByPrefixHash
// ---------------------------------------------------------------------------

TEST_F(PrefixCacheRouterTest,
       TryAcquireByPrefixHash_EmptyBlocks_ReturnsNullopt) {
  EXPECT_FALSE(router->tryAcquireByPrefixHash({}, nullptr).has_value());
}

TEST_F(PrefixCacheRouterTest, TryAcquireByPrefixHash_Miss_ReturnsNullopt) {
  EXPECT_FALSE(
      router->tryAcquireByPrefixHash(threeBlocks(), nullptr).has_value());
}

TEST_F(PrefixCacheRouterTest, TryAcquireByPrefixHash_Hit_ReturnsSession) {
  auto sessionId = addSession(7u);
  registerBlocks(sessionId, threeBlocks());

  auto acquired = router->tryAcquireByPrefixHash(threeBlocks(), nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_TRUE(acquired->sessionFound);
  EXPECT_EQ(acquired->sessionId, sessionId);
  EXPECT_EQ(acquired->slotId, 7u);
  EXPECT_GT(acquired->numberOfMatchedTokens, 0u);
}

TEST_F(PrefixCacheRouterTest,
       TryAcquireByPrefixHash_Busy_ReturnsCandidatesWithoutSession) {
  auto sessionId = addSession(8u);
  registerBlocks(sessionId, threeBlocks());

  ASSERT_TRUE(
      router->tryAcquireByPrefixHash(threeBlocks(), nullptr)->sessionFound);
  auto busy = router->tryAcquireByPrefixHash(threeBlocks(), nullptr);
  ASSERT_TRUE(busy.has_value());
  EXPECT_FALSE(busy->sessionFound);
  EXPECT_FALSE(busy->candidatesList.empty());
}

TEST_F(PrefixCacheRouterTest,
       TryAcquireByPrefixHash_StaleSession_RemovesFromIndex) {
  auto sessionId = addSession(9u, 999u);
  registerBlocks(sessionId, threeBlocks());
  sessions_[sessionId]->setHash(999u);

  auto stale = router->tryAcquireByPrefixHash(threeBlocks(), nullptr);
  ASSERT_TRUE(stale.has_value());
  EXPECT_FALSE(stale->sessionFound);

  auto sessionId2 = addSession(10u);
  registerBlocks(sessionId2, threeBlocks());
  auto hit = router->tryAcquireByPrefixHash(threeBlocks(), nullptr);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(hit->sessionId, sessionId2);
}

// ---------------------------------------------------------------------------
// tryAcquireByResponseId
// ---------------------------------------------------------------------------

TEST_F(PrefixCacheRouterTest, TryAcquireByResponseId_EmptyId_ReturnsNullopt) {
  EXPECT_FALSE(router->tryAcquireByResponseId("", nullptr).has_value());
}

TEST_F(PrefixCacheRouterTest, TryAcquireByResponseId_Miss_ReturnsNullopt) {
  EXPECT_FALSE(router->tryAcquireByResponseId("missing", nullptr).has_value());
}

TEST_F(PrefixCacheRouterTest, TryAcquireByResponseId_Hit_ReturnsSession) {
  auto sessionId = addSession(11u);
  router->registerResponseId(sessionId, "resp-1");

  auto acquired = router->tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_TRUE(acquired->sessionFound);
  EXPECT_EQ(acquired->sessionId, sessionId);
  EXPECT_EQ(acquired->slotId, 11u);
}

TEST_F(PrefixCacheRouterTest, TryAcquireByResponseId_Busy_Throws) {
  auto sessionId = addSession(12u);
  router->registerResponseId(sessionId, "resp-1");

  ASSERT_TRUE(router->tryAcquireByResponseId("resp-1", nullptr).has_value());
  EXPECT_THROW(router->tryAcquireByResponseId("resp-1", nullptr),
               std::runtime_error);
}

TEST_F(PrefixCacheRouterTest, TryAcquireByResponseId_Stale_RemovesIndexEntry) {
  auto sessionId = addSession(13u);
  router->registerResponseId(sessionId, "resp-1");
  sessions_[sessionId]->setResponseId("stale");

  EXPECT_FALSE(router->tryAcquireByResponseId("resp-1", nullptr).has_value());
  EXPECT_FALSE(router->tryAcquireByResponseId("resp-1", nullptr).has_value());
}

// ---------------------------------------------------------------------------
// register / update response id
// ---------------------------------------------------------------------------

TEST_F(PrefixCacheRouterTest, RegisterResponseId_EmptyId_IsNoOp) {
  auto sessionId = addSession(14u);
  router->registerResponseId(sessionId, "");
  EXPECT_TRUE(sessions_[sessionId]->getResponseId().empty());
}

TEST_F(PrefixCacheRouterTest, UpdateResponseId_ReKeysLookup) {
  auto sessionId = addSession(15u);
  router->registerResponseId(sessionId, "resp-1");

  router->updateResponseId("resp-1", "resp-2");
  EXPECT_FALSE(router->tryAcquireByResponseId("resp-1", nullptr).has_value());

  auto acquired = router->tryAcquireByResponseId("resp-2", nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->sessionId, sessionId);
}

// ---------------------------------------------------------------------------
// computeMatchedTokens / clearSessionBlockThinkTokens
// ---------------------------------------------------------------------------

TEST_F(PrefixCacheRouterTest, ComputeMatchedTokens_ReflectsRegisteredBlocks) {
  auto sessionId = addSession(16u);
  auto blocks = std::vector<tt::utils::BlockHashInfo>{
      {100, 5},
      {200, 12},
      {300, 20},
  };
  registerBlocks(sessionId, blocks);

  auto [matched, think] = router->computeMatchedTokens(sessionId, blocks);
  EXPECT_GT(matched, 0u);
  EXPECT_EQ(think, 20u);
}

TEST_F(PrefixCacheRouterTest, ClearSessionBlockThinkTokens_ResetsThinkCount) {
  auto sessionId = addSession(17u);
  auto blocks = std::vector<tt::utils::BlockHashInfo>{
      {100, 5},
      {200, 12},
      {300, 20},
  };
  registerBlocks(sessionId, blocks);

  router->clearSessionBlockThinkTokens(sessionId);
  auto [matched, think] = router->computeMatchedTokens(sessionId, blocks);
  EXPECT_GT(matched, 0u);
  EXPECT_EQ(think, 0u);
}

// ---------------------------------------------------------------------------
// onSessionClosed
// ---------------------------------------------------------------------------

TEST_F(PrefixCacheRouterTest, OnSessionClosed_RemovesIndexes) {
  auto sessionId = addSession(18u);
  registerBlocks(sessionId, threeBlocks());
  router->registerResponseId(sessionId, "resp-close");

  router->onSessionClosed(sessionId, 100u, "resp-close");

  EXPECT_FALSE(
      router->tryAcquireByResponseId("resp-close", nullptr).has_value());
  auto sessionId2 = addSession(19u);
  registerBlocks(sessionId2, threeBlocks());
  auto hit = router->tryAcquireByPrefixHash(threeBlocks(), nullptr);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(hit->sessionId, sessionId2);
}

// ---------------------------------------------------------------------------
// getSlot
// ---------------------------------------------------------------------------

TEST_F(PrefixCacheRouterTest, GetSlot_ResponseIdHit) {
  auto prompt = makeThreeBlockPrompt();
  auto blocks = router->computeBlockInfos(prompt);
  auto sessionId = addSession(20u);
  registerBlocks(sessionId, blocks);
  router->registerResponseId(sessionId, "resp-1");

  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "resp-1";
  opts.responseId = "resp-2";

  std::optional<tt::services::SlotAcquireResult> result;
  router->getSlot(
      prompt, std::move(opts), nullptr,
      [&](tt::services::SlotAcquireResult acquired) {
        result = std::move(acquired);
      },
      [](const std::string&) { FAIL() << "unexpected error"; });

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->sessionId, sessionId);
  EXPECT_EQ(result->slotId, 20u);
  EXPECT_FALSE(result->isNewSession);
  EXPECT_FALSE(shrunkSessions_.empty());
  EXPECT_EQ(shrunkSessions_.front().first, sessionId);

  releaseSession(sessionId);
  EXPECT_FALSE(router->tryAcquireByResponseId("resp-1", nullptr).has_value());
  EXPECT_TRUE(router->tryAcquireByResponseId("resp-2", nullptr).has_value());
}

TEST_F(PrefixCacheRouterTest, GetSlot_PrefixCacheHit) {
  auto prompt = makeThreeBlockPrompt();
  auto blocks = router->computeBlockInfos(prompt);
  auto sessionId = addSession(21u);
  registerBlocks(sessionId, blocks);

  std::optional<tt::services::SlotAcquireResult> result;
  router->getSlot(
      prompt, {}, nullptr,
      [&](tt::services::SlotAcquireResult acquired) {
        result = std::move(acquired);
      },
      [](const std::string&) { FAIL() << "unexpected error"; });

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->sessionId, sessionId);
  EXPECT_FALSE(result->isNewSession);
  releaseSession(sessionId);
}

TEST_F(PrefixCacheRouterTest, GetSlot_AllocatesNewSessionOnMiss) {
  auto prompt = makeThreeBlockPrompt();
  tt::services::GetSlotOptions opts;
  opts.responseId = "resp-new";

  std::optional<tt::services::SlotAcquireResult> result;
  router->getSlot(
      prompt, std::move(opts), nullptr,
      [&](tt::services::SlotAcquireResult acquired) {
        result = std::move(acquired);
      },
      [](const std::string&) { FAIL() << "unexpected error"; });

  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->isNewSession);
  EXPECT_EQ(result->slotId, 1000u);
  auto it = sessions_.find(result->sessionId);
  ASSERT_NE(it, sessions_.end());
  EXPECT_EQ(it->second->getResponseId(), "resp-new");
}

TEST_F(PrefixCacheRouterTest, GetSlot_BusyCandidate_CopiesFromSourceSlot) {
  auto threePrompt = makeThreeBlockPrompt();
  auto threeBlockInfos = router->computeBlockInfos(threePrompt);
  auto sessionId = addSession(22u);
  registerBlocks(sessionId, threeBlockInfos);
  ASSERT_TRUE(
      router->tryAcquireByPrefixHash(threeBlockInfos, nullptr)->sessionFound);

  auto fourPrompt = makeFourBlockPrompt();
  std::optional<tt::services::SlotAcquireResult> result;
  router->getSlot(
      fourPrompt, {}, nullptr,
      [&](tt::services::SlotAcquireResult acquired) {
        result = std::move(acquired);
      },
      [](const std::string&) { FAIL() << "unexpected error"; });

  ASSERT_TRUE(result.has_value());
  EXPECT_NE(result->sessionId, sessionId);
  EXPECT_FALSE(result->isNewSession);
  EXPECT_GT(result->matchedTokens, 0u);
  ASSERT_EQ(createdSlotCopyFrom_.size(), 1u);
  EXPECT_EQ(createdSlotCopyFrom_.front(), 22u);
  EXPECT_TRUE(lockedSlots_.empty());
}

}  // namespace
