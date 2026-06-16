// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared helpers for end-to-end tests.

#pragma once

#include <arpa/inet.h>
#include <json/json.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tt::test::e2e {

// ---------------------------------------------------------------------------
// Docker / container helpers
// ---------------------------------------------------------------------------

// Detect Docker gateway IP from /proc/net/route (for container-to-host access).
inline std::string detectDockerGateway() {
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

// ---------------------------------------------------------------------------
// Test configuration
// ---------------------------------------------------------------------------

struct E2ETestConfig {
  std::string host = detectDockerGateway();
  uint16_t port = 8080;
  std::string model = "tt-cpp-server";
  size_t firstBlockSize = 32;
  size_t blockSize = 32;

  static E2ETestConfig fromEnv() {
    E2ETestConfig cfg;
    if (const char* h = std::getenv("DYNAMO_HOST")) cfg.host = h;
    if (const char* p = std::getenv("DYNAMO_PORT")) cfg.port = std::stoi(p);
    if (const char* m = std::getenv("DYNAMO_MODEL")) cfg.model = m;
    if (const char* fb = std::getenv("KV_CACHE_FIRST_BLOCK_SIZE"))
      cfg.firstBlockSize = std::stoul(fb);
    if (const char* bs = std::getenv("KV_CACHE_BLOCK_SIZE"))
      cfg.blockSize = std::stoul(bs);
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
};

struct ChatResponse {
  int statusCode = 0;
  std::string content;
  UsageInfo usage;
  std::string error;
  bool ok() const { return statusCode == 200 && error.empty(); }
};

inline int parseStatusCode(const std::string& response) {
  // HTTP/1.1 200 OK
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

// ---------------------------------------------------------------------------
// HTTP client (with hostname resolution)
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

  std::string request = buildHttpRequest(host, port, body);
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

// ---------------------------------------------------------------------------
// JSON request building
// ---------------------------------------------------------------------------

inline std::string buildChatRequestJson(const std::string& model,
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

inline Json::Value makeMessage(const std::string& role,
                               const std::string& content) {
  Json::Value msg;
  msg["role"] = role;
  msg["content"] = content;
  return msg;
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

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

// Generate a prompt with approximately the target number of tokens.
// Uses simple repeated words to get predictable token counts.
inline std::string generatePromptWithApproxTokens(size_t targetTokens) {
  std::string msg;
  const std::vector<std::string> words = {"hello", "world", "test", "data",
                                          "check"};
  msg.reserve(targetTokens * 7);
  for (size_t i = 0; i < targetTokens; ++i) {
    msg += words[i % words.size()] + " ";
  }
  return msg;
}

// Generate a unique test ID based on current timestamp.
inline std::string generateUniqueTestId(const std::string& prefix) {
  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
  return prefix + "-" + std::to_string(millis);
}

// ---------------------------------------------------------------------------
// Prefix cache calculation
// ---------------------------------------------------------------------------

inline int computeExpectedCachedTokens(int promptTokens, size_t firstBlockSize,
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
// Rendezvous file helpers (for subprocess coordination)
// ---------------------------------------------------------------------------

inline bool writeRendezvous(const std::string& path, const std::string& data) {
  std::ofstream f(path);
  if (!f) return false;
  f << data;
  return f.good();
}

inline std::string readRendezvous(const std::string& path, int timeoutSec = 60) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream f(path);
    if (f) {
      std::string content;
      std::getline(f, content);
      if (!content.empty()) return content;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return "";
}

}  // namespace tt::test::e2e
