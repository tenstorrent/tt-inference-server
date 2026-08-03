// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared Dynamo frontend helpers for integration and E2E tests.
//
// Covers:
//   - DynamoConfig / Docker gateway detection
//   - Raw HTTP POST/GET against the Dynamo frontend
//   - SSE chat response parsing + DynamoClient
//   - Two-phase readiness: etcd backend registration, then /v1/models
//
// Integration tests keep gray-box IPC inspection in-process; they only use
// this layer to send HTTP through the external frontend. E2E binaries that
// talk to a pre-started stack reuse the same client.

#pragma once

#include <arpa/inet.h>
#include <json/json.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

#include <gtest/gtest.h>

#include "chat_request.hpp"
#include "dynamo/etcd_client.hpp"

namespace tt::test::dynamo {

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
  uint16_t port = 8080;
  // Matches deploy.sh / mock_pipeline default when unset.
  std::string model = "deepseek-ai/DeepSeek-R1-0528";

  static DynamoConfig fromEnv() {
    DynamoConfig cfg;
    if (const char* h = std::getenv("DYNAMO_HOST")) cfg.host = h;
    if (const char* p = std::getenv("DYNAMO_PORT")) cfg.port = std::stoi(p);
    if (const char* m = std::getenv("DYNAMO_MODEL")) cfg.model = m;
    return cfg;
  }
};

inline std::string etcdEndpointsFromEnv() {
  if (const char* v = std::getenv("DYNAMO_ETCD_ENDPOINTS"); v && *v) return v;
  if (const char* v = std::getenv("ETCD_ENDPOINTS"); v && *v) return v;
  return "http://127.0.0.1:2379";
}

/// Set env vars required for in-process DynamoWorkerServer registration.
/// Safe to call multiple times; does not overwrite values already set by the
/// caller / CI. Must run before TestServer::start() / settings are first read.
inline void configureDynamoEnv() {
  setenv("DYNAMO_ENDPOINT_ENABLED", "1", 0);
  setenv("DYNAMO_DISCOVERY_BACKEND", "etcd", 0);
  setenv("DYN_DISCOVERY_BACKEND", "etcd", 0);
  setenv("DYNAMO_NAMESPACE", "default", 0);
  setenv("DYNAMO_COMPONENT", "backend", 0);
  setenv("DYNAMO_ENDPOINT_NAME", "generate", 0);
  setenv("DYNAMO_BIND_HOST", "0.0.0.0", 0);

  const std::string etcd = etcdEndpointsFromEnv();
  setenv("DYNAMO_ETCD_ENDPOINTS", etcd.c_str(), 0);
  setenv("ETCD_ENDPOINTS", etcd.c_str(), 0);

  // Advertise an address the Dynamo frontend container can dial back to.
  // Prefer an explicit override; otherwise use the docker bridge gateway
  // (host side of published ports) when not already set.
  if (!std::getenv("DYN_TCP_RPC_HOST")) {
    setenv("DYN_TCP_RPC_HOST", detectDockerGateway().c_str(), 0);
  }
}

/// Prefix under which DynamoWorkerServer publishes instance keys:
///   v1/instances/<namespace>/<component>/<endpoint>/
inline std::string etcdInstancePrefixFromEnv() {
  const char* ns = std::getenv("DYNAMO_NAMESPACE");
  const char* component = std::getenv("DYNAMO_COMPONENT");
  const char* endpoint = std::getenv("DYNAMO_ENDPOINT_NAME");
  std::ostringstream oss;
  oss << "v1/instances/" << (ns && *ns ? ns : "default") << "/"
      << (component && *component ? component : "backend") << "/"
      << (endpoint && *endpoint ? endpoint : "generate") << "/";
  return oss.str();
}

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
      const auto& usage = chunk["usage"];
      result.usage.promptTokens = usage.get("prompt_tokens", 0).asInt();
      result.usage.completionTokens = usage.get("completion_tokens", 0).asInt();
      result.usage.totalTokens = usage.get("total_tokens", 0).asInt();

      if (usage.isMember("prompt_tokens_details")) {
        const auto& ptd = usage["prompt_tokens_details"];
        result.usage.cachedTokens = ptd.get("cached_tokens", 0).asInt();
      }
      if (usage.isMember("completion_tokens_details")) {
        const auto& ctd = usage["completion_tokens_details"];
        result.usage.reasoningTokens = ctd.get("reasoning_tokens", 0).asInt();
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

inline bool responseLooksComplete(const std::string& response) {
  if (response.find("data: [DONE]") != std::string::npos) return true;

  const auto headerEnd = response.find("\r\n\r\n");
  if (headerEnd == std::string::npos) return false;

  std::string headersLower = response.substr(0, headerEnd);
  for (char& c : headersLower) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  const std::string body = response.substr(headerEnd + 4);

  const std::string clKey = "content-length:";
  const auto clPos = headersLower.find(clKey);
  if (clPos == std::string::npos) return false;

  const auto lineEnd = headersLower.find("\r\n", clPos);
  const auto colon = headersLower.find(':', clPos);
  if (colon == std::string::npos) return false;
  const std::string lenStr = headersLower.substr(
      colon + 1,
      (lineEnd == std::string::npos ? headersLower.size() : lineEnd) - colon -
          1);
  try {
    const auto len = static_cast<size_t>(std::stoul(lenStr));
    return body.size() >= len;
  } catch (...) {
    return false;
  }
}

inline int connectTcp(const std::string& host, uint16_t port) {
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
  return sock;
}

inline std::string recvHttpResponse(int sock, int timeoutMs) {
  timeval tv{timeoutMs / 1000, (timeoutMs % 1000) * 1000};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  std::string response;
  char buf[4096];
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) {
    response.append(buf, static_cast<size_t>(n));
    if (responseLooksComplete(response)) break;
  }
  return response;
}

inline std::string sendHttpRequest(const std::string& host, uint16_t port,
                                   const std::string& body,
                                   int timeoutMs = 120000) {
  int sock = connectTcp(host, port);

  std::ostringstream oss;
  oss << "POST /v1/chat/completions HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "\r\n"
      << body;
  const std::string request = oss.str();
  if (::send(sock, request.c_str(), request.size(), 0) < 0) {
    ::close(sock);
    throw std::runtime_error("Failed to send request");
  }

  std::string response = recvHttpResponse(sock, timeoutMs);
  ::close(sock);
  return response;
}

inline std::string sendHttpGet(const std::string& host, uint16_t port,
                               const std::string& path,
                               int timeoutMs = 10000) {
  int sock = connectTcp(host, port);

  std::ostringstream oss;
  oss << "GET " << path << " HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n"
      << "Connection: close\r\n"
      << "\r\n";
  const std::string request = oss.str();
  if (::send(sock, request.c_str(), request.size(), 0) < 0) {
    ::close(sock);
    throw std::runtime_error("Failed to send GET " + path);
  }

  std::string response = recvHttpResponse(sock, timeoutMs);
  ::close(sock);
  return response;
}

/// Raw POST to Dynamo frontend — returns full HTTP response bytes for
/// HttpResponse::parse / ChatCompletionStream (integration gray-box path).
inline std::string sendDynamoRequest(const DynamoConfig& cfg,
                                     const std::string& body,
                                     int timeoutMs = 30000) {
  return sendHttpRequest(cfg.host, cfg.port, body, timeoutMs);
}

// ---------------------------------------------------------------------------
// JSON request building
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
// Readiness helpers
// ---------------------------------------------------------------------------

inline bool waitForTcpPort(const std::string& host, uint16_t port,
                           int timeoutSec = 60) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);

  while (std::chrono::steady_clock::now() < deadline) {
    try {
      int sock = ::socket(AF_INET, SOCK_STREAM, 0);
      if (sock < 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }

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
                                  int timeoutSec = 30) {
  return waitForTcpPort(cfg.host, cfg.port, timeoutSec);
}

/// Phase 1: poll etcd until a backend instance key appears.
inline bool waitForEtcdBackendRegistration(int timeoutSec = 30) {
  const std::string endpoints = etcdEndpointsFromEnv();
  const std::string prefix = etcdInstancePrefixFromEnv();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);

  while (std::chrono::steady_clock::now() < deadline) {
    try {
      tt::dynamo::EtcdClient client(endpoints, /*timeout_ms=*/2000);
      if (client.hasKeysWithPrefix(prefix)) {
        std::cout << "[dynamo] etcd backend registered under " << prefix
                  << std::endl;
        return true;
      }
    } catch (const std::exception& e) {
      std::cerr << "[dynamo] etcd poll failed: " << e.what() << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  std::cerr << "[dynamo] timed out waiting for etcd keys under " << prefix
            << " (etcd=" << endpoints << ")" << std::endl;
  return false;
}

inline std::optional<Json::Value> parseJsonBody(const std::string& raw) {
  const auto headerEnd = raw.find("\r\n\r\n");
  if (headerEnd == std::string::npos) return std::nullopt;
  std::string body = raw.substr(headerEnd + 4);
  Json::Value root;
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errors;
  if (!reader->parse(body.c_str(), body.c_str() + body.size(), &root,
                     &errors)) {
    return std::nullopt;
  }
  return root;
}

/// Phase 2: poll GET /v1/models until `cfg.model` appears (etcd watch lag).
inline bool waitForModelDiscovery(const DynamoConfig& cfg,
                                  int timeoutSec = 30) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);

  while (std::chrono::steady_clock::now() < deadline) {
    try {
      const std::string raw =
          sendHttpGet(cfg.host, cfg.port, "/v1/models", /*timeoutMs=*/5000);
      if (parseStatusCode(raw) != 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        continue;
      }
      auto json = parseJsonBody(raw);
      if (!json || !(*json).isMember("data") || !(*json)["data"].isArray()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        continue;
      }
      for (const auto& m : (*json)["data"]) {
        if (m.isMember("id") && m["id"].asString() == cfg.model) {
          std::cout << "[dynamo] model " << cfg.model
                    << " visible in /v1/models" << std::endl;
          return true;
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "[dynamo] /v1/models poll failed: " << e.what() << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  std::cerr << "[dynamo] timed out waiting for model " << cfg.model
            << " at " << cfg.host << ":" << cfg.port << std::endl;
  return false;
}

// ---------------------------------------------------------------------------
// DynamoClient - high-level chat API
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

inline std::string generateUniqueTestId(const std::string& prefix) {
  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
  return prefix + "-" + std::to_string(millis);
}

// ---------------------------------------------------------------------------
// DynamoTestFixture - shared suite setup for integration / disagg E2E
// ---------------------------------------------------------------------------
//
// Usage:
//   class MyTest : public DynamoTestFixture<MyTest> {
//    protected:
//     static void SetUpTestSuite() {
//       if (!initDynamo()) return;
//       // start in-process backend...
//       if (!waitUntilBackendRoutable()) return;
//     }
//   };
//
// initDynamo() fails fast if the frontend TCP port is unreachable.
// waitUntilBackendRoutable() is the two-phase check after the backend
// registers: etcd key appears, then /v1/models lists the model.

template <typename Derived>
class DynamoTestFixture : public ::testing::Test {
 protected:
  /// Phase 0: frontend must already be up (deploy.sh --no-worker).
  static bool initDynamo(int timeoutSec = 30) {
    dynamoConfig_ = DynamoConfig::fromEnv();
    if (!waitForDynamoFrontend(dynamoConfig_, timeoutSec)) {
      dynamoAvailable_ = false;
      dynamoUnavailableReason_ =
          "Dynamo frontend not reachable at " + dynamoConfig_.host + ":" +
          std::to_string(dynamoConfig_.port) +
          ". Start with: cd dynamo_frontend && ./deploy.sh --no-monitoring "
          "--no-worker";
      std::cerr << "[" << testName() << "] " << dynamoUnavailableReason_
                << std::endl;
      return false;
    }
    dynamoAvailable_ = true;
    std::cout << "[" << testName() << "] Dynamo frontend ready at "
              << dynamoConfig_.host << ":" << dynamoConfig_.port << std::endl;
    return true;
  }

  /// Phases 1–2: after in-process DynamoWorkerServer::start(), wait until the
  /// frontend can route to it.
  static bool waitUntilBackendRoutable(int timeoutSec = 30) {
    if (!waitForEtcdBackendRegistration(timeoutSec)) {
      dynamoAvailable_ = false;
      dynamoUnavailableReason_ =
          "No cpp_server backend registered in etcd under " +
          etcdInstancePrefixFromEnv();
      std::cerr << "[" << testName() << "] " << dynamoUnavailableReason_
                << std::endl;
      return false;
    }
    if (!waitForModelDiscovery(dynamoConfig_, timeoutSec)) {
      dynamoAvailable_ = false;
      dynamoUnavailableReason_ =
          "Model " + dynamoConfig_.model +
          " not visible in /v1/models after backend registration";
      std::cerr << "[" << testName() << "] " << dynamoUnavailableReason_
                << std::endl;
      return false;
    }
    return true;
  }

  void SetUp() override {
    if (!dynamoAvailable_) {
      FAIL() << dynamoUnavailableReason_;
    }
  }

  static std::future<std::string> asyncRequest(const std::string& body,
                                               int timeoutMs = 60000) {
    return std::async(std::launch::async, [body, timeoutMs] {
      return sendDynamoRequest(dynamoConfig_, body, timeoutMs);
    });
  }

  static std::future<std::string> asyncRequest(const tt::test::ChatRequest& req,
                                               int timeoutMs = 60000) {
    return asyncRequest(req.toJson(), timeoutMs);
  }

  static tt::test::ChatRequest chatRequest() {
    return tt::test::ChatRequest().model(dynamoConfig_.model);
  }

  static const DynamoConfig& dynamoConfig() { return dynamoConfig_; }
  static bool dynamoAvailable() { return dynamoAvailable_; }

  static DynamoConfig dynamoConfig_;
  static bool dynamoAvailable_;
  static std::string dynamoUnavailableReason_;

 private:
  static const char* testName() { return typeid(Derived).name(); }
};

template <typename Derived>
DynamoConfig DynamoTestFixture<Derived>::dynamoConfig_;

template <typename Derived>
bool DynamoTestFixture<Derived>::dynamoAvailable_ = false;

template <typename Derived>
std::string DynamoTestFixture<Derived>::dynamoUnavailableReason_ =
    "Dynamo infrastructure not initialized";

}  // namespace tt::test::dynamo
