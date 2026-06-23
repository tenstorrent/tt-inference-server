// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// End-to-end prefix cache verification via Dynamo frontend.
//
// This test connects to an external Dynamo frontend server. Set environment
// variables to configure:
//   DYNAMO_HOST (default: 127.0.0.1)
//   DYNAMO_PORT (default: 8080)
//   DYNAMO_MODEL (default: tt-cpp-server)
//
// The test expects Dynamo to be running. Start it with:
//   cd dynamo_frontend && ./deploy.sh --local-build
//
// Usage:
//   ./prefix_cache_e2e_test

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "config/defaults.hpp"
#include "support/dynamo_test_fixture.hpp"

namespace {

using namespace tt::test;

// The mock_pipeline backend emits a fixed reasoning sequence:
//   <think> + 10 content tokens + </think> = 12 reasoning tokens.
// The prefix hash excludes these tokens from the hashable completion count.
// NOTE: The Dynamo frontend currently does not forward
// completion_tokens_details (including reasoning_tokens) from the backend, so
// we cannot read this value dynamically. Once Dynamo is patched to pass through
// completion_tokens_details, this constant can be replaced with
// usage.reasoningTokens.
constexpr int K_MOCK_REASONING_TOKENS = 12;

int computeExpectedCachedTokens(int promptTokens, size_t firstBlockSize,
                                size_t blockSize) {
  if (promptTokens < static_cast<int>(firstBlockSize)) {
    return 0;
  }

  int cached = static_cast<int>(firstBlockSize);
  int remaining = promptTokens - static_cast<int>(firstBlockSize);
  int fullSubsequentBlocks = remaining / static_cast<int>(blockSize);
  cached += fullSubsequentBlocks * static_cast<int>(blockSize);
  return cached;
}

// ---------------------------------------------------------------------------
// Test configuration (extends DynamoConfig with KV cache settings)
// ---------------------------------------------------------------------------

struct PrefixCacheTestConfig {
  DynamoConfig dynamo;
  size_t firstBlockSize = tt::config::defaults::KV_CACHE_FIRST_BLOCK_SIZE;
  size_t blockSize = tt::config::defaults::KV_CACHE_BLOCK_SIZE;

  static PrefixCacheTestConfig fromEnv() {
    PrefixCacheTestConfig cfg;
    cfg.dynamo = DynamoConfig::fromEnv();
    cfg.dynamo.model = "deepseek-ai/DeepSeek-R1-0528";
    if (const char* m = std::getenv("DYNAMO_MODEL")) cfg.dynamo.model = m;
    if (const char* fb = std::getenv("KV_CACHE_FIRST_BLOCK_SIZE"))
      cfg.firstBlockSize = std::stoul(fb);
    if (const char* bs = std::getenv("KV_CACHE_BLOCK_SIZE"))
      cfg.blockSize = std::stoul(bs);
    return cfg;
  }
};

// ---------------------------------------------------------------------------
// System prompts that fill at least one hash block (~200+ tokens).
// Each prompt starts with completely different text to ensure no prefix overlap
// between tests. The unique suffix added at runtime ensures no overlap with
// prior test runs.
// ---------------------------------------------------------------------------

const char* kSystemPromptCoding =
    "CODING ASSISTANT ALPHA: You are a highly capable AI coding assistant "
    "working inside an IDE. You have access to the full project source tree "
    "and can read files, search code, run shell commands, and edit files. "
    "When the user asks you to make changes, you should understand the "
    "request, explore the relevant code, plan your changes, and implement "
    "them carefully. Always verify your changes compile and pass tests. If "
    "you encounter ambiguity, ask for clarification. Provide concise, "
    "accurate answers. When writing code, follow the project's existing style "
    "and conventions. Use meaningful variable names and add comments only "
    "when the logic is non-obvious. Keep functions small and focused. Prefer "
    "composition over inheritance. Handle errors gracefully and log relevant "
    "context. When reviewing code, look for correctness issues first, then "
    "performance, then style. Always consider edge cases and concurrent "
    "access patterns. For distributed systems, think about failure modes, "
    "retry strategies, and idempotency. When working with databases, consider "
    "indexing, query plans, and data migration paths. For API design, follow "
    "RESTful conventions and provide clear error messages with appropriate "
    "HTTP status codes.";

const char* kSystemPromptMarine =
    "MARINE BIOLOGY EXPERT: You specialize in oceanography and marine life. "
    "Your expertise covers coral reef ecosystems, deep sea creatures, whale "
    "migration patterns, and the impact of climate change on ocean health. "
    "When discussing marine topics, always consider the interconnected nature "
    "of ocean systems. Explain concepts clearly for both scientists and "
    "general audiences. Reference recent research when applicable. Discuss "
    "conservation efforts and their effectiveness. Consider both local and "
    "global scales of ocean phenomena. Address the relationship between human "
    "activities and marine ecosystem health. Provide specific examples from "
    "different ocean regions including the Pacific, Atlantic, Indian Ocean, "
    "and polar waters. Discuss the role of phytoplankton in carbon "
    "sequestration. Explain trophic cascades and keystone species in marine "
    "environments. Consider seasonal variations in marine productivity. "
    "Address invasive species and their impacts on native marine communities. "
    "Discuss marine protected areas and their effectiveness in conservation. "
    "Consider the economic importance of fisheries and sustainable harvesting "
    "practices.";

const char* kSystemPromptAstronomy =
    "ASTRONOMY SPECIALIST: You are an expert in astrophysics and space "
    "exploration. Your knowledge spans stellar evolution, galaxy formation, "
    "black holes, exoplanets, and the search for extraterrestrial life. When "
    "discussing astronomical topics, explain complex phenomena in accessible "
    "terms while maintaining scientific accuracy. Reference current missions "
    "and discoveries from NASA, ESA, and other space agencies. Discuss both "
    "observational techniques and theoretical frameworks. Consider the scale "
    "of cosmic distances and timescales. Address the history of astronomical "
    "discovery and how our understanding has evolved. Explain the tools "
    "astronomers use including telescopes, spectroscopy, and gravitational "
    "wave detectors. Discuss the cosmic microwave background and what it "
    "tells us about the early universe. Consider dark matter and dark energy "
    "and their role in cosmic structure. Address planetary formation and the "
    "conditions necessary for habitable worlds. Discuss the life cycles of "
    "stars from protostars to remnants.";

}  // namespace

class PrefixCacheE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    cfg = PrefixCacheTestConfig::fromEnv();
    client = std::make_unique<DynamoClient>(cfg.dynamo);

    std::cout << "Prefix cache E2E test against " << cfg.dynamo.host << ":"
              << cfg.dynamo.port << std::endl;
    std::cout << "  model=" << cfg.dynamo.model
              << "  firstBlockSize=" << cfg.firstBlockSize
              << "  blockSize=" << cfg.blockSize << std::endl;
    std::cout << "Waiting for server..." << std::endl;

    ASSERT_TRUE(client->waitForServer()) << "Server not ready within timeout";
    std::cout << "Server ready, warming up..." << std::endl;

    ASSERT_TRUE(client->warmup())
        << "Server warmup failed (Dynamo frontend may not have discovered "
           "backends)";
    std::cout << "Server warmed up." << std::endl;
  }

  static void TearDownTestSuite() { client.reset(); }

  ChatResponse sendChat(const std::vector<Json::Value>& messages,
                        int maxTokens = 32) {
    return client->sendChat(messages, maxTokens);
  }

  static PrefixCacheTestConfig cfg;
  static std::unique_ptr<DynamoClient> client;
};

PrefixCacheTestConfig PrefixCacheE2ETest::cfg;
std::unique_ptr<DynamoClient> PrefixCacheE2ETest::client;

TEST_F(PrefixCacheE2ETest, CacheReplayScenario) {
  // Test cache behavior across multiple requests including replays.
  //
  // 1. Request 1: Fresh prompt -> cached_tokens = 0
  // 2. Request 2: Continuation -> cached_tokens grows
  // 3. Request 3: Different prompt -> cached_tokens = 0
  // 4. Request 4: Replay R1 -> cached_tokens = block-aligned(R1 prompt)
  // 5. Request 5: Replay R2 -> cached_tokens = block-aligned(R2 prompt)

  std::cout << "\n=== Test: Cache replay scenario ===" << std::endl;

  std::string uniqueSuffix = " [" + generateUniqueTestId("replay-test") + "]";

  // Request 1: Fresh prompt (coding assistant theme)
  std::vector<Json::Value> r1Messages = {
      makeMessage("system", std::string(kSystemPromptCoding) + uniqueSuffix),
      makeMessage("user", "What is the capital of France?")};

  std::cout << "  Request 1 (fresh prompt)..." << std::endl;
  ChatResponse r1 = sendChat(r1Messages);
  ASSERT_TRUE(r1.ok()) << "Request 1 failed: " << r1.error;
  std::cout << "    prompt=" << r1.usage.promptTokens
            << " cached=" << r1.usage.cachedTokens
            << " completion=" << r1.usage.completionTokens << std::endl;

  EXPECT_EQ(r1.usage.cachedTokens, 0)
      << "Request 1 should have cached_tokens=0 (fresh unique prompt)";

  std::vector<Json::Value> r1MessagesCopy = r1Messages;

  // Request 2: Continuation of R1
  std::vector<Json::Value> r2Messages = r1Messages;
  r2Messages.push_back(makeMessage("assistant", r1.content));
  r2Messages.push_back(
      makeMessage("user", "And what is the capital of Germany?"));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 2 (continuation of R1)..." << std::endl;
  ChatResponse r2 = sendChat(r2Messages);
  ASSERT_TRUE(r2.ok()) << "Request 2 failed: " << r2.error;
  std::cout << "    prompt=" << r2.usage.promptTokens
            << " cached=" << r2.usage.cachedTokens
            << " completion=" << r2.usage.completionTokens << std::endl;

  // R2's prompt includes the assistant response as text. With mock_pipeline,
  // the tokenized assistant text round-trips to the same tokens as R1's
  // completion, so R2 matches R1's full session. The prefix hash excludes
  // thinking tokens, so we use (completion - reasoning) for the hashable part.
  // Use K_MOCK_REASONING_TOKENS since Dynamo doesn't forward reasoning_tokens
  // yet.
  int r1HashableCompletion =
      r1.usage.completionTokens - K_MOCK_REASONING_TOKENS;
  int r1SessionTokens = r1.usage.promptTokens + r1HashableCompletion;
  int r2ExpectedCached = computeExpectedCachedTokens(
      r1SessionTokens, cfg.firstBlockSize, cfg.blockSize);
  std::cout << "    Expected cached: " << r2ExpectedCached << std::endl;

  EXPECT_GT(r2.usage.cachedTokens, 0)
      << "Request 2 should have cached_tokens > 0 (reuses R1's cached prefix)";
  EXPECT_LE(std::abs(r2.usage.cachedTokens - r2ExpectedCached), 1)
      << "Request 2 cached should match block-aligned R1 session";

  std::vector<Json::Value> r2MessagesCopy = r2Messages;

  // Request 3: Different prompt (marine biology theme, no prefix overlap)
  std::vector<Json::Value> r3Messages = {
      makeMessage("system", std::string(kSystemPromptMarine) + uniqueSuffix),
      makeMessage("user", "Tell me about coral reef ecosystems.")};

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 3 (different prompt)..." << std::endl;
  ChatResponse r3 = sendChat(r3Messages);
  ASSERT_TRUE(r3.ok()) << "Request 3 failed: " << r3.error;
  std::cout << "    prompt=" << r3.usage.promptTokens
            << " cached=" << r3.usage.cachedTokens
            << " completion=" << r3.usage.completionTokens << std::endl;

  EXPECT_EQ(r3.usage.cachedTokens, 0)
      << "Request 3 should have cached_tokens=0 (completely different prompt)";

  // Request 4: Replay exact R1 prompt -> should hit full cache
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 4 (replay R1 prompt)..." << std::endl;
  ChatResponse r4 = sendChat(r1MessagesCopy);
  ASSERT_TRUE(r4.ok()) << "Request 4 failed: " << r4.error;
  std::cout << "    prompt=" << r4.usage.promptTokens
            << " cached=" << r4.usage.cachedTokens
            << " completion=" << r4.usage.completionTokens << std::endl;

  int r4ExpectedCached = computeExpectedCachedTokens(
      r4.usage.promptTokens, cfg.firstBlockSize, cfg.blockSize);
  std::cout << "    Expected cached: " << r4ExpectedCached << std::endl;

  EXPECT_GT(r4.usage.cachedTokens, 0)
      << "Request 4 should hit cache (replaying R1 prompt)";
  EXPECT_LE(std::abs(r4.usage.cachedTokens - r4ExpectedCached), 1)
      << "Request 4 cached should match block-aligned R4 prompt";

  // Request 5: Replay exact R2 prompt -> should hit full cache
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 5 (replay R2 prompt)..." << std::endl;
  ChatResponse r5 = sendChat(r2MessagesCopy);
  ASSERT_TRUE(r5.ok()) << "Request 5 failed: " << r5.error;
  std::cout << "    prompt=" << r5.usage.promptTokens
            << " cached=" << r5.usage.cachedTokens
            << " completion=" << r5.usage.completionTokens << std::endl;

  int r5ExpectedCached = computeExpectedCachedTokens(
      r1SessionTokens, cfg.firstBlockSize, cfg.blockSize);
  std::cout << "    Expected cached: " << r5ExpectedCached << std::endl;

  EXPECT_GT(r5.usage.cachedTokens, 0)
      << "Request 5 should hit cache (replaying R2 prompt)";
  EXPECT_LE(std::abs(r5.usage.cachedTokens - r5ExpectedCached), 1)
      << "Request 5 cached should match block-aligned R1 session";

  std::cout << "  OK: All replay scenarios passed" << std::endl;
}

TEST_F(PrefixCacheE2ETest, MultiTurnHashCreation) {
  // Test that multi-turn conversations create proper prefix hashes.
  // Simulates guideLLM multi-turn scenario: each turn builds on the previous,
  // and prefix cache should hit on the shared history.

  std::cout << "\n=== Test: Multi-turn hash creation ===" << std::endl;

  std::string uniquePrefix =
      "[" + generateUniqueTestId("MULTITURN-TEST") + "] ";

  // Turn 1
  std::vector<Json::Value> messages = {
      makeMessage("system", uniquePrefix + std::string(kSystemPromptCoding)),
      makeMessage("user", "What is a hash table?")};

  std::cout << "  Turn 1..." << std::endl;
  ChatResponse t1 = sendChat(messages);
  ASSERT_TRUE(t1.ok()) << "Turn 1 failed: " << t1.error;
  std::cout << "    prompt=" << t1.usage.promptTokens
            << " cached=" << t1.usage.cachedTokens << std::endl;

  EXPECT_EQ(t1.usage.cachedTokens, 0) << "Turn 1 should have cached=0 (fresh)";
  int t1Prompt = t1.usage.promptTokens;

  std::cout << "t1.content: " << t1.content << std::endl;

  // Turn 2
  messages.push_back(makeMessage("assistant", t1.content));
  messages.push_back(makeMessage("user", "How does it handle collisions?"));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Turn 2..." << std::endl;
  ChatResponse t2 = sendChat(messages);
  ASSERT_TRUE(t2.ok()) << "Turn 2 failed: " << t2.error;
  std::cout << "    prompt=" << t2.usage.promptTokens
            << " cached=" << t2.usage.cachedTokens << std::endl;

  // Turn 2 should hit cache on turn 1's hashable session state. The prefix
  // hash excludes thinking tokens, so we compute expected cached based on
  // prompt + (completion - reasoning) tokens.
  // Use K_MOCK_REASONING_TOKENS since Dynamo doesn't forward reasoning_tokens
  // yet.
  int t1HashableCompletion =
      t1.usage.completionTokens - K_MOCK_REASONING_TOKENS;
  int t1SessionTokens = t1Prompt + t1HashableCompletion;
  int t2ExpectedCached = computeExpectedCachedTokens(
      t1SessionTokens, cfg.firstBlockSize, cfg.blockSize);
  std::cout << "    t1 session tokens (hashable): " << t1SessionTokens
            << " (prompt=" << t1Prompt
            << " + completion=" << t1HashableCompletion << " ["
            << t1.usage.completionTokens << " - " << K_MOCK_REASONING_TOKENS
            << " mock reasoning])" << std::endl;
  std::cout << "    Expected cached: " << t2ExpectedCached << std::endl;
  EXPECT_GT(t2.usage.cachedTokens, 0) << "Turn 2 should hit prefix cache";
  EXPECT_LE(std::abs(t2.usage.cachedTokens - t2ExpectedCached), 1)
      << "Turn 2 cached should match block-aligned turn 1 session";

  std::cout << "t2.content: " << t2.content << std::endl;

  // Turn 3
  messages.push_back(makeMessage("assistant", t2.content));
  messages.push_back(makeMessage("user", "What about open addressing?"));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Turn 3..." << std::endl;
  ChatResponse t3 = sendChat(messages);
  ASSERT_TRUE(t3.ok()) << "Turn 3 failed: " << t3.error;
  std::cout << "    prompt=" << t3.usage.promptTokens
            << " cached=" << t3.usage.cachedTokens << std::endl;

  // Turn 3 should hit cache on turn 2's hashable session state. The prefix
  // hash excludes thinking tokens, so we compute expected cached based on
  // prompt + (completion - reasoning) tokens.
  // Use K_MOCK_REASONING_TOKENS since Dynamo doesn't forward reasoning_tokens
  // yet.
  int t2HashableCompletion =
      t2.usage.completionTokens - K_MOCK_REASONING_TOKENS;
  int t2SessionTokens = t2.usage.promptTokens + t2HashableCompletion;
  int t3ExpectedCached = computeExpectedCachedTokens(
      t2SessionTokens, cfg.firstBlockSize, cfg.blockSize);
  std::cout << "    t2 session tokens (hashable): " << t2SessionTokens
            << " (prompt=" << t2.usage.promptTokens
            << " + completion=" << t2HashableCompletion << " ["
            << t2.usage.completionTokens << " - " << K_MOCK_REASONING_TOKENS
            << " mock reasoning])" << std::endl;
  std::cout << "    Expected cached: " << t3ExpectedCached << std::endl;
  EXPECT_GT(t3.usage.cachedTokens, 0) << "Turn 3 should hit prefix cache";
  EXPECT_LE(std::abs(t3.usage.cachedTokens - t3ExpectedCached), 1)
      << "Turn 3 cached should match block-aligned turn 2 session";

  std::cout << "  OK: Multi-turn hash creation working" << std::endl;
}

TEST_F(PrefixCacheE2ETest, SessionEvictionUnderLoad) {
  // Test that eviction policy works when many sessions are created.
  // Create multiple unique conversations to fill up session slots.
  // Verify that prefix cache still works after eviction occurs.

  std::cout << "\n=== Test: Session eviction under load ===" << std::endl;

  int64_t millis = currentTimeMillis();
  constexpr int kNumConversations = 20;

  std::cout << "  Creating " << kNumConversations << " unique conversations..."
            << std::endl;

  for (int i = 0; i < kNumConversations; ++i) {
    std::string uniquePrefix = "[EVICTION-TEST-" + std::to_string(millis) +
                               "-CONV-" + std::to_string(i) + "] ";

    std::vector<Json::Value> messages = {
        makeMessage("system", uniquePrefix + std::string(kSystemPromptCoding)),
        makeMessage("user", "Question " + std::to_string(i))};

    ChatResponse r = sendChat(messages);
    ASSERT_TRUE(r.ok()) << "Conversation " << i << " failed: " << r.error;

    EXPECT_EQ(r.usage.cachedTokens, 0)
        << "Conversation " << i << " should have cached=0 (unique prompt)";
  }

  std::cout << "  All " << kNumConversations << " conversations created"
            << std::endl;

  // Now create a conversation and verify prefix caching still works
  std::string finalPrefix =
      "[EVICTION-TEST-" + std::to_string(millis) + "-FINAL] ";

  std::vector<Json::Value> finalMessages = {
      makeMessage("system", finalPrefix + std::string(kSystemPromptMarine)),
      makeMessage("user", "Tell me about whales.")};

  std::cout << "  Final conversation - request 1 (fresh)..." << std::endl;
  ChatResponse f1 = sendChat(finalMessages);
  ASSERT_TRUE(f1.ok()) << "Final request 1 failed: " << f1.error;
  std::cout << "    prompt=" << f1.usage.promptTokens
            << " cached=" << f1.usage.cachedTokens << std::endl;

  EXPECT_EQ(f1.usage.cachedTokens, 0) << "Final R1 should have cached=0";

  // Replay the same prompt - should hit cache
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Final conversation - request 2 (replay)..." << std::endl;
  ChatResponse f2 = sendChat(finalMessages);
  ASSERT_TRUE(f2.ok()) << "Final request 2 failed: " << f2.error;
  std::cout << "    prompt=" << f2.usage.promptTokens
            << " cached=" << f2.usage.cachedTokens << std::endl;

  int f2ExpectedCached = computeExpectedCachedTokens(
      f2.usage.promptTokens, cfg.firstBlockSize, cfg.blockSize);
  EXPECT_GT(f2.usage.cachedTokens, 0)
      << "Final R2 should hit cache (system still functional after load)";
  EXPECT_LE(std::abs(f2.usage.cachedTokens - f2ExpectedCached), 1)
      << "Final R2 cached should match expected";

  std::cout << "  OK: System functional after load, prefix cache working"
            << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
