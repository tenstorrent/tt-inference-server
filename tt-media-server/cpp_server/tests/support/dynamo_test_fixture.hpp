// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared test fixture and helpers for Dynamo-based integration/E2E tests.
//
// Provides:
//   - DynamoConfig: configuration for Dynamo frontend connection
//   - DynamoTestFixture: base fixture with availability checking and graceful
//   skip
//   - DynamoClient: high-level chat API for E2E tests
//   - HTTP client helpers for sending requests via Dynamo frontend
//   - SSE response parsing utilities
//   - ChatRequest fluent builder (see chat_request.hpp)
//
// Usage (integration tests with fixture):
//   class MyTest : public DynamoTestFixture<MyTest> {
//    protected:
//     static void SetUpTestSuite() {
//       if (!initDynamo()) return;
//       // Your test-specific setup here...
//     }
//   };
//
// Usage (E2E tests with DynamoClient):
//   DynamoConfig cfg = DynamoConfig::fromEnv();
//   DynamoClient client(cfg);
//   client.waitForServer();
//   client.warmup();
//   auto response = client.sendChat({makeMessage("user", "hello")});

#pragma once

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <json/json.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "chat_request.hpp"

namespace tt::test {

// ---------------------------------------------------------------------------
// Docker / container helpers
// ---------------------------------------------------------------------------

inline std::string detectDockerGateway() {
  std::ifstream route("/proc/net/route");
  if (!route) return "127.0.0.1";

  std::string line;
  std::getline(route, line);  // skip header
  while (std::getline(route, line)) {
    std::istringstream iss(line);
    std::string iface, dest, gateway;
    if (iss >> iface >> dest >> gateway && dest == "00000000") {
      unsigned int gw = std::stoul(gateway, nullptr, 16);
      unsigned char* bytes = reinterpret_cast<unsigned char*>(&gw);
      return std::to_string(bytes[0]) + "." + std::to_string(bytes[1]) + "." +
             std::to_string(bytes[2]) + "." + std::to_string(bytes[3]);
    }
  }
  return "127.0.0.1";
}

// ---------------------------------------------------------------------------
// Dynamo configuration
// ---------------------------------------------------------------------------

struct DynamoConfig {
  std::string host = detectDockerGateway();
  uint16_t port = 8080;  // Dynamo frontend default from deploy.sh
  // Default model matches deploy.sh default
  std::string model = "deepseek-ai/DeepSeek-R1-0528";

  static DynamoConfig fromEnv() {
    DynamoConfig cfg;
    if (const char* h = std::getenv("DYNAMO_HOST")) cfg.host = h;
    if (const char* p = std::getenv("DYNAMO_PORT")) cfg.port = std::stoi(p);
    if (const char* m = std::getenv("DYNAMO_MODEL")) cfg.model = m;
    return cfg;
  }
};

// ---------------------------------------------------------------------------
// SSE response parsing
// ---------------------------------------------------------------------------

struct UsageInfo {
  int promptTokens = 0;
  int completionTokens = 0;
  int totalTokens = 0;
  int cachedTokens = 0;
  int reasoningTokens = 0;  // Thinking tokens excluded from prefix hash
};

struct ChatResponse {
  int statusCode = 0;
  std::string content;
  UsageInfo usage;
  std::string error;
  bool ok() const { return statusCode == 200 && error.empty(); }
};

inline int parseStatusCode(const std::string& response) {
  auto pos = response.find(' ');
  if (pos == std::string::npos) return 0;
  return std::stoi(response.substr(pos + 1, 3));
}

inline ChatResponse parseStreamingResponse(const std::string& rawResponse) {
  ChatResponse result;
  result.statusCode = parseStatusCode(rawResponse);

  if (result.statusCode != 200) {
    result.error = "HTTP " + std::to_string(result.statusCode);
    return result;
  }

  std::istringstream stream(rawResponse);
  std::string line;

  while (std::getline(stream, line)) {
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

    if (chunk.isMember("usage") && chunk["usage"].isObject()) {
      const auto& usageVal = chunk["usage"];
      result.usage.promptTokens = usageVal.get("prompt_tokens", 0).asInt();
      result.usage.completionTokens =
          usageVal.get("completion_tokens", 0).asInt();
      result.usage.totalTokens = usageVal.get("total_tokens", 0).asInt();

      if (usageVal.isMember("prompt_tokens_details")) {
        const auto& ptd = usageVal["prompt_tokens_details"];
        result.usage.cachedTokens = ptd.get("cached_tokens", 0).asInt();
      }
      if (usageVal.isMember("completion_tokens_details")) {
        const auto& ctd = usageVal["completion_tokens_details"];
        result.usage.reasoningTokens = ctd.get("reasoning_tokens", 0).asInt();
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// HTTP client helpers
// ---------------------------------------------------------------------------

inline std::string buildHttpRequest(const std::string& host, uint16_t port,
                                    const std::string& body) {
  std::ostringstream oss;
  oss << "POST /v1/chat/completions HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "\r\n"
      << body;
  return oss.str();
}

inline std::string sendHttpRequest(const std::string& host, uint16_t port,
                                   const std::string& body,
                                   int timeoutMs = 120000) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    throw std::runtime_error("Failed to create socket");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
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

  std::string request = buildHttpRequest(host, port, body);
  if (::send(sock, request.c_str(), request.size(), 0) < 0) {
    ::close(sock);
    throw std::runtime_error("Failed to send request");
  }

  timeval tv{timeoutMs / 1000, (timeoutMs % 1000) * 1000};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  std::string response;
  char buf[4096];
  ssize_t n;

  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) {
    response.append(buf, static_cast<size_t>(n));
    if (response.find("data: [DONE]") != std::string::npos) {
      break;
    }
  }

  ::close(sock);
  return response;
}

inline std::string sendDynamoRequest(const DynamoConfig& cfg,
                                     const std::string& body,
                                     int timeoutMs = 30000) {
  return sendHttpRequest(cfg.host, cfg.port, body, timeoutMs);
}

// ---------------------------------------------------------------------------
// Server readiness helpers
// ---------------------------------------------------------------------------

inline bool waitForTcpPort(const std::string& host, uint16_t port,
                           int timeoutSec = 60) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);

  while (std::chrono::steady_clock::now() < deadline) {
    try {
      int sock = ::socket(AF_INET, SOCK_STREAM, 0);
      if (sock < 0) continue;

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

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

inline bool waitForDynamoFrontend(const DynamoConfig& cfg,
                                  int timeoutSec = 60) {
  return waitForTcpPort(cfg.host, cfg.port, timeoutSec);
}

inline bool warmupDynamoFrontend(const DynamoConfig& cfg) {
  const std::string warmupBody =
      R"({"model":")" + cfg.model +
      R"(","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true})";

  for (int attempt = 0; attempt < 10; ++attempt) {
    try {
      std::string response =
          sendHttpRequest(cfg.host, cfg.port, warmupBody, 30000);
      if (response.find("HTTP/1.1 200") != std::string::npos ||
          response.find("data:") != std::string::npos) {
        return true;
      }
    } catch (...) {
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  return false;
}

// ---------------------------------------------------------------------------
// JSON request building helpers
// ---------------------------------------------------------------------------

inline std::string buildChatRequestJson(
    const std::string& model, const std::vector<Json::Value>& messages,
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

inline Json::Value makeMessage(const std::string& role,
                               const std::string& content) {
  Json::Value msg;
  msg["role"] = role;
  msg["content"] = content;
  return msg;
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

inline std::string generateUniqueTestId(const std::string& prefix) {
  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
  return prefix + "-" + std::to_string(millis);
}

inline int64_t currentTimeMillis() {
  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
}

// ---------------------------------------------------------------------------
// Dynamo environment configuration
// ---------------------------------------------------------------------------

inline void configureDynamoEnv() {
  setenv("DYNAMO_ENDPOINT_ENABLED", "1", 1);
  setenv("DYNAMO_BIND_HOST", "0.0.0.0", 1);
  setenv("DYNAMO_NAMESPACE", "default", 1);
  setenv("DYNAMO_COMPONENT", "backend", 1);
  setenv("DYNAMO_ENDPOINT_NAME", "generate", 1);
  setenv("DYNAMO_ETCD_ENDPOINTS", "http://127.0.0.1:2379", 1);
  setenv("DYNAMO_ETCD_LEASE_TTL_SECS", "30", 1);
}

// ---------------------------------------------------------------------------
// DynamoClient - high-level chat API for E2E tests
// ---------------------------------------------------------------------------

class DynamoClient {
 public:
  explicit DynamoClient(const DynamoConfig& cfg) : cfg_(cfg) {}

  ChatResponse sendChat(const std::vector<Json::Value>& messages,
                        int maxTokens = 32) {
    std::string body =
        buildChatRequestJson(cfg_.model, messages, maxTokens, true);
    try {
      std::string rawResponse = sendHttpRequest(cfg_.host, cfg_.port, body);
      return parseStreamingResponse(rawResponse);
    } catch (const std::exception& e) {
      ChatResponse result;
      result.error = e.what();
      return result;
    }
  }

  bool warmup(int maxAttempts = 5) {
    std::vector<Json::Value> warmupMessages = {
        makeMessage("system", "You are a helpful assistant."),
        makeMessage("user", "Say hello.")};
    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
      ChatResponse r = sendChat(warmupMessages, 8);
      if (r.ok()) {
        std::cout << "Warmup succeeded after " << (attempt + 1) << " attempt(s)"
                  << std::endl;
        return true;
      }
      std::cout << "Warmup attempt " << (attempt + 1) << " failed: " << r.error
                << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    return false;
  }

  bool waitForServer(int timeoutSec = 60) {
    return waitForTcpPort(cfg_.host, cfg_.port, timeoutSec);
  }

  const DynamoConfig& config() const { return cfg_; }

 private:
  DynamoConfig cfg_;
};

// ---------------------------------------------------------------------------
// DynamoTestFixture - base fixture for integration tests
// ---------------------------------------------------------------------------

template <typename Derived>
class DynamoTestFixture : public ::testing::Test {
 protected:
  static bool initDynamo(int timeoutSec = 5) {
    dynamoConfig_ = DynamoConfig::fromEnv();
    if (!waitForDynamoFrontend(dynamoConfig_, timeoutSec)) {
      dynamoAvailable_ = false;
      std::cerr << "[" << testName() << "] Dynamo frontend not available at "
                << dynamoConfig_.host << ":" << dynamoConfig_.port << std::endl;
      std::cerr << "  Start with: cd dynamo_frontend && ./deploy.sh "
                   "--local-build"
                << std::endl;
      return false;
    }
    dynamoAvailable_ = true;
    return true;
  }

  static bool warmupDynamo() {
    if (!warmupDynamoFrontend(dynamoConfig_)) {
      std::cerr << "[" << testName() << "] Dynamo frontend warmup failed "
                << "(backend may not have registered with etcd)" << std::endl;
      dynamoAvailable_ = false;
      return false;
    }
    return true;
  }

  void SetUp() override {
    if (!dynamoAvailable_) {
      GTEST_SKIP() << "Dynamo frontend not available. Start with: "
                   << "cd dynamo_frontend && ./deploy.sh --local-build";
    }
  }

  static std::future<std::string> asyncRequest(const std::string& body,
                                               int timeoutMs = 30000) {
    return std::async(std::launch::async, [body, timeoutMs] {
      return sendDynamoRequest(dynamoConfig_, body, timeoutMs);
    });
  }

  static std::future<std::string> asyncRequest(const ChatRequest& req,
                                               int timeoutMs = 30000) {
    return asyncRequest(req.toJson(), timeoutMs);
  }

  static ChatRequest chatRequest() {
    return ChatRequest().model(dynamoConfig_.model);
  }

  static const DynamoConfig& dynamoConfig() { return dynamoConfig_; }
  static bool dynamoAvailable() { return dynamoAvailable_; }

  static DynamoConfig dynamoConfig_;
  static bool dynamoAvailable_;

 private:
  static const char* testName() { return typeid(Derived).name(); }
};

template <typename Derived>
DynamoConfig DynamoTestFixture<Derived>::dynamoConfig_;

template <typename Derived>
bool DynamoTestFixture<Derived>::dynamoAvailable_ = false;

}  // namespace tt::test
