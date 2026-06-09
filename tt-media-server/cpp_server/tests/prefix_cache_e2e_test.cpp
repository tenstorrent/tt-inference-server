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

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <json/json.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "config/defaults.hpp"

namespace {

// Detect Docker gateway IP from /proc/net/route (for container-to-host access)
std::string detectDockerGateway() {
  std::ifstream route("/proc/net/route");
  if (!route) return "127.0.0.1";

  std::string line;
  std::getline(route, line);  // skip header
  while (std::getline(route, line)) {
    std::istringstream iss(line);
    std::string iface, dest, gateway;
    if (iss >> iface >> dest >> gateway && dest == "00000000") {
      // Gateway is hex-encoded little-endian IP
      unsigned int gw = std::stoul(gateway, nullptr, 16);
      unsigned char* bytes = reinterpret_cast<unsigned char*>(&gw);
      return std::to_string(bytes[0]) + "." + std::to_string(bytes[1]) + "." +
             std::to_string(bytes[2]) + "." + std::to_string(bytes[3]);
    }
  }
  return "127.0.0.1";
}

struct TestConfig {
  std::string host = detectDockerGateway();
  uint16_t port = 8080;  // Dynamo frontend default from deploy.sh
  std::string model = "deepseek-ai/DeepSeek-R1-0528";
  std::string apiKey = "your-secret-key";
  size_t firstBlockSize = tt::config::defaults::KV_CACHE_FIRST_BLOCK_SIZE;
  size_t blockSize = tt::config::defaults::KV_CACHE_BLOCK_SIZE;

  static TestConfig fromEnv() {
    TestConfig cfg;
    if (const char* h = std::getenv("DYNAMO_HOST")) cfg.host = h;
    if (const char* p = std::getenv("DYNAMO_PORT")) cfg.port = std::stoi(p);
    if (const char* m = std::getenv("DYNAMO_MODEL")) cfg.model = m;
    if (const char* k = std::getenv("OPENAI_API_KEY")) cfg.apiKey = k;
    if (const char* fb = std::getenv("KV_CACHE_FIRST_BLOCK_SIZE"))
      cfg.firstBlockSize = std::stoul(fb);
    if (const char* bs = std::getenv("KV_CACHE_BLOCK_SIZE"))
      cfg.blockSize = std::stoul(bs);
    return cfg;
  }
};

struct UsageInfo {
  int promptTokens = 0;
  int completionTokens = 0;
  int totalTokens = 0;
  int cachedTokens = 0;
};

struct ChatResponse {
  int statusCode = 0;
  std::string content;
  UsageInfo usage;
  std::string error;
  bool ok() const { return statusCode == 200 && error.empty(); }
};

std::string buildHttpRequest(const std::string& host, uint16_t port,
                             const std::string& body,
                             const std::string& apiKey) {
  std::ostringstream oss;
  oss << "POST /v1/chat/completions HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Authorization: Bearer " << apiKey << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "\r\n"
      << body;
  return oss.str();
}

std::string sendHttpRequest(const std::string& host, uint16_t port,
                            const std::string& body, const std::string& apiKey,
                            int timeoutMs = 120000) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    throw std::runtime_error("Failed to create socket");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
    // Try hostname resolution
    struct hostent* he = ::gethostbyname(host.c_str());
    if (!he) {
      ::close(sock);
      throw std::runtime_error("Failed to resolve host: " + host);
    }
    std::memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
  }

  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(sock);
    throw std::runtime_error("Failed to connect to " + host + ":" +
                             std::to_string(port));
  }

  std::string request = buildHttpRequest(host, port, body, apiKey);
  if (::send(sock, request.c_str(), request.size(), 0) < 0) {
    ::close(sock);
    throw std::runtime_error("Failed to send request");
  }

  // Set receive timeout
  timeval tv{timeoutMs / 1000, (timeoutMs % 1000) * 1000};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  std::string response;
  char buf[4096];
  ssize_t n;

  // For SSE, we read until we see "data: [DONE]" or timeout
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) {
    response.append(buf, static_cast<size_t>(n));
    if (response.find("data: [DONE]") != std::string::npos) {
      break;
    }
  }

  ::close(sock);
  return response;
}

int parseStatusCode(const std::string& response) {
  // HTTP/1.1 200 OK
  auto pos = response.find(' ');
  if (pos == std::string::npos) return 0;
  return std::stoi(response.substr(pos + 1, 3));
}

ChatResponse parseStreamingResponse(const std::string& rawResponse) {
  ChatResponse result;
  result.statusCode = parseStatusCode(rawResponse);

  if (result.statusCode != 200) {
    result.error = "HTTP " + std::to_string(result.statusCode);
    return result;
  }

  // Parse SSE events
  std::istringstream stream(rawResponse);
  std::string line;

  while (std::getline(stream, line)) {
    // Remove trailing \r if present
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (line.rfind("data: ", 0) != 0) continue;

    std::string data = line.substr(6);
    if (data == "[DONE]") break;

    Json::Value chunk;
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    std::string errors;

    if (!reader->parse(data.c_str(), data.c_str() + data.size(), &chunk,
                       &errors)) {
      continue;
    }

    // Extract content delta (DeepSeek R1 uses reasoning_content, not content)
    if (chunk.isMember("choices") && chunk["choices"].isArray() &&
        !chunk["choices"].empty()) {
      const auto& delta = chunk["choices"][0]["delta"];
      if (delta.isMember("content") && !delta["content"].isNull()) {
        result.content += delta["content"].asString();
      }
      if (delta.isMember("reasoning_content") &&
          !delta["reasoning_content"].isNull()) {
        result.content += delta["reasoning_content"].asString();
      }
    }

    // Extract usage (last chunk)
    if (chunk.isMember("usage") && chunk["usage"].isObject()) {
      const auto& usage = chunk["usage"];
      result.usage.promptTokens = usage.get("prompt_tokens", 0).asInt();
      result.usage.completionTokens = usage.get("completion_tokens", 0).asInt();
      result.usage.totalTokens = usage.get("total_tokens", 0).asInt();

      if (usage.isMember("prompt_tokens_details")) {
        const auto& ptd = usage["prompt_tokens_details"];
        result.usage.cachedTokens = ptd.get("cached_tokens", 0).asInt();
      }
    }
  }

  return result;
}

std::string buildChatRequestJson(const std::string& model,
                                 const std::vector<Json::Value>& messages,
                                 int maxTokens = 32, bool stream = true) {
  Json::Value root;
  root["model"] = model;
  root["max_tokens"] = maxTokens;
  root["stream"] = stream;

  if (stream) {
    Json::Value streamOptions;
    streamOptions["include_usage"] = true;
    root["stream_options"] = streamOptions;
  }

  Json::Value messagesArray(Json::arrayValue);
  for (const auto& msg : messages) {
    messagesArray.append(msg);
  }
  root["messages"] = messagesArray;

  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  return Json::writeString(writer, root);
}

Json::Value makeMessage(const std::string& role, const std::string& content) {
  Json::Value msg;
  msg["role"] = role;
  msg["content"] = content;
  return msg;
}

ChatResponse sendChat(const TestConfig& cfg,
                      const std::vector<Json::Value>& messages,
                      int maxTokens = 32) {
  std::string body = buildChatRequestJson(cfg.model, messages, maxTokens);
  try {
    std::string rawResponse =
        sendHttpRequest(cfg.host, cfg.port, body, cfg.apiKey);
    return parseStreamingResponse(rawResponse);
  } catch (const std::exception& e) {
    ChatResponse result;
    result.error = e.what();
    return result;
  }
}

bool waitForServer(const TestConfig& cfg, int timeoutSec = 60) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);

  while (std::chrono::steady_clock::now() < deadline) {
    try {
      int sock = ::socket(AF_INET, SOCK_STREAM, 0);
      if (sock < 0) continue;

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(cfg.port);
      ::inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr);

      timeval tv{2, 0};
      ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

      bool connected = ::connect(sock, reinterpret_cast<sockaddr*>(&addr),
                                 sizeof(addr)) == 0;
      ::close(sock);

      if (connected) return true;
    } catch (...) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return false;
}

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

// System prompts that fill at least one hash block (~200+ tokens).
// Each prompt starts with completely different text to ensure no prefix overlap
// between tests. The unique suffix added at runtime ensures no overlap with
// prior test runs.

// Test 1: Coding assistant theme
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

// Test 2: Marine biology theme (completely different prefix)
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

// Test 3: Astronomy theme (completely different prefix)
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
    cfg = TestConfig::fromEnv();
    std::cout << "Prefix cache E2E test against " << cfg.host << ":" << cfg.port
              << std::endl;
    std::cout << "  model=" << cfg.model
              << "  firstBlockSize=" << cfg.firstBlockSize
              << "  blockSize=" << cfg.blockSize << std::endl;
    std::cout << "Waiting for server..." << std::endl;

    ASSERT_TRUE(waitForServer(cfg)) << "Server not ready within timeout";
    std::cout << "Server ready." << std::endl;
  }

  static TestConfig cfg;
};

TestConfig PrefixCacheE2ETest::cfg;

TEST_F(PrefixCacheE2ETest, CacheReplayScenario) {
  // Test cache behavior across multiple requests including replays.
  //
  // 1. Request 1: Fresh prompt → cached_tokens = 0
  // 2. Request 2: Continuation → cached_tokens grows
  // 3. Request 3: Different prompt → cached_tokens = 0
  // 4. Request 4: Replay R1 → cached_tokens = block-aligned(R1 prompt)
  // 5. Request 5: Replay R2 → cached_tokens = block-aligned(R2 prompt)

  std::cout << "\n=== Test: Cache replay scenario ===" << std::endl;

  // Unique suffix ensures no overlap with prior test runs
  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
  std::string uniqueSuffix = " [replay-test-" + std::to_string(millis) + "]";

  // -------------------------------------------------------------------------
  // Request 1: Fresh prompt (coding assistant theme)
  // -------------------------------------------------------------------------
  std::vector<Json::Value> r1Messages = {
      makeMessage("system", std::string(kSystemPromptCoding) + uniqueSuffix),
      makeMessage("user", "What is the capital of France?")};

  std::cout << "  Request 1 (fresh prompt)..." << std::endl;
  ChatResponse r1 = sendChat(cfg, r1Messages);
  ASSERT_TRUE(r1.ok()) << "Request 1 failed: " << r1.error;
  std::cout << "    prompt=" << r1.usage.promptTokens
            << " cached=" << r1.usage.cachedTokens
            << " completion=" << r1.usage.completionTokens << std::endl;

  EXPECT_EQ(r1.usage.cachedTokens, 0)
      << "Request 1 should have cached_tokens=0 (fresh unique prompt)";

  std::vector<Json::Value> r1MessagesCopy = r1Messages;

  // -------------------------------------------------------------------------
  // Request 2: Continuation of R1
  // -------------------------------------------------------------------------
  std::vector<Json::Value> r2Messages = r1Messages;
  r2Messages.push_back(makeMessage("assistant", r1.content));
  r2Messages.push_back(
      makeMessage("user", "And what is the capital of Germany?"));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 2 (continuation of R1)..." << std::endl;
  ChatResponse r2 = sendChat(cfg, r2Messages);
  ASSERT_TRUE(r2.ok()) << "Request 2 failed: " << r2.error;
  std::cout << "    prompt=" << r2.usage.promptTokens
            << " cached=" << r2.usage.cachedTokens
            << " completion=" << r2.usage.completionTokens << std::endl;

  // R2's prompt includes the assistant response as text, but re-tokenization
  // produces different token IDs than R1's original completion. So R2 only
  // matches R1's prompt prefix.
  int r2ExpectedCached = computeExpectedCachedTokens(
      r1.usage.promptTokens, cfg.firstBlockSize, cfg.blockSize);
  std::cout << "    Expected cached: " << r2ExpectedCached << std::endl;

  EXPECT_GT(r2.usage.cachedTokens, 0)
      << "Request 2 should have cached_tokens > 0 (reuses R1's cached prefix)";
  EXPECT_LE(std::abs(r2.usage.cachedTokens - r2ExpectedCached), 1)
      << "Request 2 cached should match block-aligned R1 prompt (matching "
         "prefix)";

  // Save R2 messages for replay later
  std::vector<Json::Value> r2MessagesCopy = r2Messages;
  int r2CachedTokens = r2.usage.cachedTokens;

  // -------------------------------------------------------------------------
  // Request 3: Different prompt (marine biology theme, no prefix overlap)
  // -------------------------------------------------------------------------
  std::vector<Json::Value> r3Messages = {
      makeMessage("system", std::string(kSystemPromptMarine) + uniqueSuffix),
      makeMessage("user", "Tell me about coral reef ecosystems.")};

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 3 (different prompt)..." << std::endl;
  ChatResponse r3 = sendChat(cfg, r3Messages);
  ASSERT_TRUE(r3.ok()) << "Request 3 failed: " << r3.error;
  std::cout << "    prompt=" << r3.usage.promptTokens
            << " cached=" << r3.usage.cachedTokens
            << " completion=" << r3.usage.completionTokens << std::endl;

  EXPECT_EQ(r3.usage.cachedTokens, 0)
      << "Request 3 should have cached_tokens=0 (completely different prompt)";

  // -------------------------------------------------------------------------
  // Request 4: Replay exact R1 prompt → should hit full cache
  // -------------------------------------------------------------------------
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 4 (replay R1 prompt)..." << std::endl;
  ChatResponse r4 = sendChat(cfg, r1MessagesCopy);
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
      << "Request 4 cached should match block-aligned R1 prompt";

  // -------------------------------------------------------------------------
  // Request 5: Replay exact R2 prompt → should hit full cache
  // -------------------------------------------------------------------------
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Request 5 (replay R2 prompt)..." << std::endl;
  ChatResponse r5 = sendChat(cfg, r2MessagesCopy);
  ASSERT_TRUE(r5.ok()) << "Request 5 failed: " << r5.error;
  std::cout << "    prompt=" << r5.usage.promptTokens
            << " cached=" << r5.usage.cachedTokens
            << " completion=" << r5.usage.completionTokens << std::endl;

  // R5 replays R2 exactly. Like R2, it only matches R1's prompt prefix
  // because re-tokenization of the assistant text diverges from cached tokens.
  int r5ExpectedCached = computeExpectedCachedTokens(
      r1.usage.promptTokens, cfg.firstBlockSize, cfg.blockSize);
  std::cout << "    Expected cached: " << r5ExpectedCached << std::endl;

  EXPECT_LE(std::abs(r5.usage.cachedTokens - r5ExpectedCached), 1)
      << "Request 5 cached should match block-aligned R1 prompt (matching "
         "prefix)";

  EXPECT_EQ(r5.usage.cachedTokens, r2CachedTokens)
      << "Request 5 (replay) should have same cached_tokens as original R2";

  std::cout << "  OK: All replay scenarios passed" << std::endl;
}

TEST_F(PrefixCacheE2ETest, MultiTurnHashCreation) {
  // Test that multi-turn conversations create proper prefix hashes.
  // Simulates guideLLM multi-turn scenario: each turn builds on the previous,
  // and prefix cache should hit on the shared history.
  //
  // Turn 1: system + user1 → cached = 0
  // Turn 2: system + user1 + assistant1 + user2 → cached > 0 (matches turn 1
  // prefix) Turn 3: system + user1 + assistant1 + user2 + assistant2 + user3 →
  // cached > turn2

  std::cout << "\n=== Test: Multi-turn hash creation ===" << std::endl;

  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
  // Prepend unique prefix to change the first block hash (prefix cache hashes
  // from the start, so appending at the end doesn't prevent cache hits)
  std::string uniquePrefix = "[MULTITURN-TEST-" + std::to_string(millis) + "] ";

  // Turn 1
  std::vector<Json::Value> messages = {
      makeMessage("system", uniquePrefix + std::string(kSystemPromptCoding)),
      makeMessage("user", "What is a hash table?")};

  std::cout << "  Turn 1..." << std::endl;
  ChatResponse t1 = sendChat(cfg, messages);
  ASSERT_TRUE(t1.ok()) << "Turn 1 failed: " << t1.error;
  std::cout << "    prompt=" << t1.usage.promptTokens
            << " cached=" << t1.usage.cachedTokens << std::endl;

  EXPECT_EQ(t1.usage.cachedTokens, 0) << "Turn 1 should have cached=0 (fresh)";
  int t1Prompt = t1.usage.promptTokens;

  // Turn 2
  messages.push_back(makeMessage("assistant", t1.content));
  messages.push_back(makeMessage("user", "How does it handle collisions?"));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Turn 2..." << std::endl;
  ChatResponse t2 = sendChat(cfg, messages);
  ASSERT_TRUE(t2.ok()) << "Turn 2 failed: " << t2.error;
  std::cout << "    prompt=" << t2.usage.promptTokens
            << " cached=" << t2.usage.cachedTokens << std::endl;

  // Turn 2 should hit cache on turn 1's prefix
  int t2ExpectedCached =
      computeExpectedCachedTokens(t1Prompt, cfg.firstBlockSize, cfg.blockSize);
  EXPECT_GT(t2.usage.cachedTokens, 0) << "Turn 2 should hit prefix cache";
  EXPECT_LE(std::abs(t2.usage.cachedTokens - t2ExpectedCached), 1)
      << "Turn 2 cached should match block-aligned turn 1 prompt";

  int t2Prompt = t2.usage.promptTokens;

  // Turn 3
  messages.push_back(makeMessage("assistant", t2.content));
  messages.push_back(makeMessage("user", "What about open addressing?"));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Turn 3..." << std::endl;
  ChatResponse t3 = sendChat(cfg, messages);
  ASSERT_TRUE(t3.ok()) << "Turn 3 failed: " << t3.error;
  std::cout << "    prompt=" << t3.usage.promptTokens
            << " cached=" << t3.usage.cachedTokens << std::endl;

  // Turn 3 should hit cache on turn 2's prompt prefix (block-aligned)
  int t3ExpectedCached =
      computeExpectedCachedTokens(t2Prompt, cfg.firstBlockSize, cfg.blockSize);
  EXPECT_GT(t3.usage.cachedTokens, 0) << "Turn 3 should hit prefix cache";
  EXPECT_LE(std::abs(t3.usage.cachedTokens - t3ExpectedCached), 1)
      << "Turn 3 cached should match block-aligned turn 2 prompt";

  std::cout << "  OK: Multi-turn hash creation working" << std::endl;
}

TEST_F(PrefixCacheE2ETest, SessionEvictionUnderLoad) {
  // Test that eviction policy works when many sessions are created.
  // Create multiple unique conversations to fill up session slots.
  // Verify that prefix cache still works after eviction occurs.
  //
  // Note: This test doesn't directly observe eviction (no API for that),
  // but it verifies the system remains functional under load and
  // prefix caching continues to work after many sessions are created.

  std::cout << "\n=== Test: Session eviction under load ===" << std::endl;

  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();

  // Create many unique conversations to trigger potential eviction
  // Each conversation has a unique system prompt prefix to change
  // the first block hash and prevent cache hits between conversations
  constexpr int kNumConversations = 20;

  std::cout << "  Creating " << kNumConversations << " unique conversations..."
            << std::endl;

  for (int i = 0; i < kNumConversations; ++i) {
    std::string uniquePrefix = "[EVICTION-TEST-" + std::to_string(millis) +
                               "-CONV-" + std::to_string(i) + "] ";

    std::vector<Json::Value> messages = {
        makeMessage("system", uniquePrefix + std::string(kSystemPromptCoding)),
        makeMessage("user", "Question " + std::to_string(i))};

    ChatResponse r = sendChat(cfg, messages);
    ASSERT_TRUE(r.ok()) << "Conversation " << i << " failed: " << r.error;

    // First request for each unique conversation should have cached=0
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
  ChatResponse f1 = sendChat(cfg, finalMessages);
  ASSERT_TRUE(f1.ok()) << "Final request 1 failed: " << f1.error;
  std::cout << "    prompt=" << f1.usage.promptTokens
            << " cached=" << f1.usage.cachedTokens << std::endl;

  EXPECT_EQ(f1.usage.cachedTokens, 0) << "Final R1 should have cached=0";

  // Replay the same prompt - should hit cache
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "  Final conversation - request 2 (replay)..." << std::endl;
  ChatResponse f2 = sendChat(cfg, finalMessages);
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
