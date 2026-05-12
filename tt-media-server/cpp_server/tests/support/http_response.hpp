// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Lightweight HTTP/1.1 response parser for the integration tests. Wraps the
// raw bytes returned by tt::test::sendAndReceive so tests can assert on
// status code, headers, JSON body, or SSE event stream — instead of
// hand-rolled string searches.
//
//   auto response = HttpResponse::parse(sendAndReceive(...));
//   EXPECT_EQ(response.statusCode(), 200);
//   EXPECT_EQ(response.json()["choices"].size(), 1u);

#pragma once

#include <json/json.h>

#include <cctype>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tt::test {

class HttpResponse {
 public:
  static HttpResponse parse(const std::string& raw) {
    HttpResponse r;

    // Status line: "HTTP/1.1 <code> <reason>\r\n"
    const auto eolStatus = raw.find("\r\n");
    if (eolStatus == std::string::npos)
      throw std::runtime_error("HttpResponse: no status line");
    const auto sp1 = raw.find(' ');
    const auto sp2 = raw.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp1 >= eolStatus)
      throw std::runtime_error("HttpResponse: malformed status line");
    r.statusCode_ = std::stoi(raw.substr(sp1 + 1, sp2 - sp1 - 1));

    // Headers: "<Name>: <Value>\r\n" lines, terminated by "\r\n\r\n"
    const auto bodyStart = raw.find("\r\n\r\n", eolStatus);
    if (bodyStart == std::string::npos)
      throw std::runtime_error("HttpResponse: no header/body separator");

    std::istringstream hs(raw.substr(eolStatus + 2, bodyStart - eolStatus - 2));
    std::string line;
    while (std::getline(hs, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      auto value = line.substr(colon + 1);
      const auto vstart = value.find_first_not_of(" \t");
      value = vstart == std::string::npos ? "" : value.substr(vstart);
      r.headers_[lowercase(line.substr(0, colon))] = std::move(value);
    }

    auto rawBody = raw.substr(bodyStart + 4);
    r.body_ = r.header("transfer-encoding").find("chunked") != std::string::npos
                  ? decodeChunked(rawBody)
                  : std::move(rawBody);
    return r;
  }

  int statusCode() const { return statusCode_; }
  const std::string& body() const { return body_; }

  // Case-insensitive header lookup. Empty string if absent.
  std::string header(std::string_view name) const {
    const auto it = headers_.find(lowercase(name));
    return it == headers_.end() ? "" : it->second;
  }

  // Parse body as JSON. Throws on parse error.
  Json::Value json() const {
    Json::Value root;
    Json::CharReaderBuilder b;
    std::string errs;
    std::istringstream is(body_);
    if (!Json::parseFromStream(b, is, &root, &errs))
      throw std::runtime_error("HttpResponse::json: parse failed: " + errs);
    return root;
  }

  // Split SSE body into "data: ..." payloads (without the "data: " prefix).
  // Useful for /v1/chat/completions stream=true.
  std::vector<std::string> sseEvents() const {
    std::vector<std::string> events;
    std::istringstream is(body_);
    std::string line;
    while (std::getline(is, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.rfind("data: ", 0) == 0) events.push_back(line.substr(6));
    }
    return events;
  }

 private:
  static std::string lowercase(std::string_view s) {
    std::string out(s);
    for (char& c : out)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
  }

  static std::string decodeChunked(std::string_view raw) {
    std::string out;
    size_t pos = 0;
    while (pos < raw.size()) {
      const auto eol = raw.find("\r\n", pos);
      if (eol == std::string_view::npos) break;
      auto sizeHex = raw.substr(pos, eol - pos);
      const auto semi = sizeHex.find(';');
      if (semi != std::string_view::npos) sizeHex = sizeHex.substr(0, semi);
      const size_t sz = std::stoul(std::string(sizeHex), nullptr, 16);
      if (sz == 0) break;
      out.append(raw.substr(eol + 2, sz));
      pos = eol + 2 + sz + 2;  // chunk-size CRLF data CRLF
    }
    return out;
  }

  int statusCode_ = 0;
  std::map<std::string, std::string> headers_;  // names lowercased
  std::string body_;
};

}  // namespace tt::test
