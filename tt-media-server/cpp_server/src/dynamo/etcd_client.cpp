// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/etcd_client.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <json/json.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <string>

namespace tt::dynamo {

namespace {

// ---------------------------------------------------------------------------
// Base64 (RFC 4648, standard alphabet, padding). etcd's JSON gateway expects
// keys/values to be base64-encoded strings.
// ---------------------------------------------------------------------------

constexpr char K_BASE64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64Encode(const std::string& in) {
  std::string out;
  out.reserve(((in.size() + 2) / 3) * 4);
  size_t i = 0;
  while (i + 3 <= in.size()) {
    uint32_t v = (static_cast<uint8_t>(in[i]) << 16) |
                 (static_cast<uint8_t>(in[i + 1]) << 8) |
                 static_cast<uint8_t>(in[i + 2]);
    out += K_BASE64_ALPHABET[(v >> 18) & 0x3F];
    out += K_BASE64_ALPHABET[(v >> 12) & 0x3F];
    out += K_BASE64_ALPHABET[(v >> 6) & 0x3F];
    out += K_BASE64_ALPHABET[v & 0x3F];
    i += 3;
  }
  const size_t rem = in.size() - i;
  if (rem == 1) {
    uint32_t v = static_cast<uint8_t>(in[i]) << 16;
    out += K_BASE64_ALPHABET[(v >> 18) & 0x3F];
    out += K_BASE64_ALPHABET[(v >> 12) & 0x3F];
    out += "==";
  } else if (rem == 2) {
    uint32_t v = (static_cast<uint8_t>(in[i]) << 16) |
                 (static_cast<uint8_t>(in[i + 1]) << 8);
    out += K_BASE64_ALPHABET[(v >> 18) & 0x3F];
    out += K_BASE64_ALPHABET[(v >> 12) & 0x3F];
    out += K_BASE64_ALPHABET[(v >> 6) & 0x3F];
    out += "=";
  }
  return out;
}

// ---------------------------------------------------------------------------
// URL parsing: accept "http://host:port" optionally with trailing "/" and
// optionally a comma-separated list (we keep the first endpoint).
// ---------------------------------------------------------------------------

struct ParsedUrl {
  std::string host;
  int port;
};

ParsedUrl parseEtcdUrl(const std::string& urlList) {
  std::string url = urlList;
  if (auto comma = url.find(','); comma != std::string::npos) {
    url = url.substr(0, comma);
  }
  // strip leading whitespace
  while (!url.empty() &&
         std::isspace(static_cast<unsigned char>(url.front()))) {
    url.erase(url.begin());
  }
  if (url.rfind("https://", 0) == 0) {
    throw EtcdError(
        "EtcdClient: HTTPS endpoints are not supported (use a TLS-terminating "
        "sidecar)");
  }
  if (url.rfind("http://", 0) == 0) {
    url.erase(0, 7);
  }
  if (auto slash = url.find('/'); slash != std::string::npos) {
    url.resize(slash);
  }
  ParsedUrl out;
  if (auto colon = url.find(':'); colon != std::string::npos) {
    out.host = url.substr(0, colon);
    try {
      out.port = std::stoi(url.substr(colon + 1));
    } catch (...) {
      throw EtcdError("EtcdClient: invalid port in endpoint '" + urlList + "'");
    }
  } else {
    out.host = url;
    out.port = 2379;
  }
  if (out.host.empty()) {
    throw EtcdError("EtcdClient: empty host in endpoint '" + urlList + "'");
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tiny blocking HTTP/1.1 client. Connect with a deadline (so etcd-not-running
// fails fast), POST a JSON body, parse Content-Length-bounded response.
// ---------------------------------------------------------------------------

class Socket {
 public:
  Socket() = default;
  ~Socket() {
    if (fd >= 0) ::close(fd);
  }
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;

  int getFd() const { return fd; }
  void reset(int newFd) {
    if (fd >= 0) ::close(fd);
    fd = newFd;
  }

 private:
  int fd = -1;
};

void connectWithTimeout(int fd, const sockaddr* addr, socklen_t len,
                        int timeoutMs) {
  int flags = ::fcntl(fd, F_GETFL, 0);
  ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

  int rc = ::connect(fd, addr, len);
  if (rc == 0) {
    ::fcntl(fd, F_SETFL, flags);
    return;
  }
  if (errno != EINPROGRESS) {
    throw EtcdError(std::string("EtcdClient: connect failed: ") +
                    std::strerror(errno));
  }
  pollfd pfd{fd, POLLOUT, 0};
  int pr = ::poll(&pfd, 1, timeoutMs);
  if (pr <= 0) {
    throw EtcdError("EtcdClient: connect timeout");
  }
  int err = 0;
  socklen_t errlen = sizeof(err);
  if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &errlen) < 0 || err != 0) {
    throw EtcdError(std::string("EtcdClient: connect error: ") +
                    std::strerror(err));
  }
  ::fcntl(fd, F_SETFL, flags);
}

void writeAll(int fd, const std::string& data, int timeoutMs) {
  size_t sent = 0;
  while (sent < data.size()) {
    pollfd pfd{fd, POLLOUT, 0};
    int pr = ::poll(&pfd, 1, timeoutMs);
    if (pr <= 0) throw EtcdError("EtcdClient: send timeout");
    ssize_t n =
        ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      throw EtcdError(std::string("EtcdClient: send failed: ") +
                      std::strerror(errno));
    }
    sent += static_cast<size_t>(n);
  }
}

/// Read bytes until either `until` appears in the buffer or `maxBytes` is
/// hit. Used to find the end of HTTP headers ("\r\n\r\n").
std::string readUntil(int fd, const std::string& until, size_t maxBytes,
                      int timeoutMs) {
  std::string buf;
  buf.reserve(1024);
  while (buf.find(until) == std::string::npos) {
    if (buf.size() >= maxBytes) {
      throw EtcdError("EtcdClient: response headers too large");
    }
    pollfd pfd{fd, POLLIN, 0};
    int pr = ::poll(&pfd, 1, timeoutMs);
    if (pr <= 0) throw EtcdError("EtcdClient: read timeout (headers)");
    char tmp[1024];
    ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n == 0) throw EtcdError("EtcdClient: connection closed in headers");
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      throw EtcdError(std::string("EtcdClient: recv failed: ") +
                      std::strerror(errno));
    }
    buf.append(tmp, static_cast<size_t>(n));
  }
  return buf;
}

std::string readExact(int fd, size_t n, int timeoutMs,
                      const std::string& alreadyRead) {
  std::string buf = alreadyRead;
  buf.reserve(n);
  while (buf.size() < n) {
    pollfd pfd{fd, POLLIN, 0};
    int pr = ::poll(&pfd, 1, timeoutMs);
    if (pr <= 0) throw EtcdError("EtcdClient: read timeout (body)");
    char tmp[4096];
    size_t want = std::min(sizeof(tmp), n - buf.size());
    ssize_t r = ::recv(fd, tmp, want, 0);
    if (r == 0) throw EtcdError("EtcdClient: connection closed in body");
    if (r < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      throw EtcdError(std::string("EtcdClient: recv failed: ") +
                      std::strerror(errno));
    }
    buf.append(tmp, static_cast<size_t>(r));
  }
  return buf;
}

/// Case-insensitive header lookup.
bool findHeader(const std::string& headers, const char* name,
                std::string* value) {
  std::string lower = headers;
  for (auto& c : lower) c = static_cast<char>(std::tolower(c));
  std::string key = std::string("\r\n") + name + ":";
  for (auto& c : key) c = static_cast<char>(std::tolower(c));
  auto pos = lower.find(key);
  if (pos == std::string::npos) return false;
  pos += key.size();
  auto end = lower.find("\r\n", pos);
  if (end == std::string::npos) return false;
  std::string raw = headers.substr(pos, end - pos);
  while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.front())))
    raw.erase(raw.begin());
  while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back())))
    raw.pop_back();
  *value = raw;
  return true;
}

/// POST `body` (assumed JSON) to `path` and return the response body. Raises
/// EtcdError on transport failure or a non-2xx status.
std::string httpPostJson(const std::string& host, int port,
                         const std::string& path, const std::string& body,
                         int timeoutMs) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  std::string portStr = std::to_string(port);
  int gai = ::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &res);
  if (gai != 0 || res == nullptr) {
    throw EtcdError(std::string("EtcdClient: getaddrinfo(") + host +
                    ") failed: " + ::gai_strerror(gai));
  }
  Socket sock;
  EtcdError lastErr("EtcdClient: no addresses tried");
  bool connected = false;
  for (auto* p = res; p != nullptr; p = p->ai_next) {
    int fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    try {
      connectWithTimeout(fd, p->ai_addr, p->ai_addrlen, timeoutMs);
      sock.reset(fd);
      connected = true;
      break;
    } catch (const EtcdError& e) {
      ::close(fd);
      lastErr = e;
    }
  }
  ::freeaddrinfo(res);
  if (!connected) throw lastErr;

  std::ostringstream req;
  req << "POST " << path << " HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n"
      << "\r\n"
      << body;
  writeAll(sock.getFd(), req.str(), timeoutMs);

  // Read headers up to "\r\n\r\n", then determine body length.
  std::string head = readUntil(sock.getFd(), "\r\n\r\n", 32 * 1024, timeoutMs);
  auto headerEnd = head.find("\r\n\r\n");
  std::string headers = head.substr(0, headerEnd + 2);
  std::string already = head.substr(headerEnd + 4);

  // status line: HTTP/1.1 <code> <reason>
  int status = 0;
  {
    auto firstSpace = headers.find(' ');
    if (firstSpace == std::string::npos) {
      throw EtcdError("EtcdClient: malformed response status line");
    }
    auto secondSpace = headers.find(' ', firstSpace + 1);
    if (secondSpace == std::string::npos) {
      throw EtcdError("EtcdClient: malformed response status line");
    }
    try {
      status = std::stoi(
          headers.substr(firstSpace + 1, secondSpace - firstSpace - 1));
    } catch (...) {
      throw EtcdError("EtcdClient: malformed response status code");
    }
  }

  std::string bodyOut;
  std::string contentLength;
  std::string transferEncoding;
  bool chunked = false;
  if (findHeader(headers, "Transfer-Encoding", &transferEncoding)) {
    chunked = (transferEncoding.find("chunked") != std::string::npos);
  }
  if (chunked) {
    // Minimal chunked decoder: read chunk-size lines and the bytes that
    // follow until a zero-sized chunk arrives.
    std::string buf = std::move(already);
    auto readMore = [&]() {
      pollfd pfd{sock.getFd(), POLLIN, 0};
      int pr = ::poll(&pfd, 1, timeoutMs);
      if (pr <= 0) throw EtcdError("EtcdClient: read timeout (chunked)");
      char tmp[4096];
      ssize_t r = ::recv(sock.getFd(), tmp, sizeof(tmp), 0);
      if (r == 0) throw EtcdError("EtcdClient: closed mid chunk");
      if (r < 0) throw EtcdError("EtcdClient: recv chunked failed");
      buf.append(tmp, static_cast<size_t>(r));
    };
    while (true) {
      while (buf.find("\r\n") == std::string::npos) readMore();
      auto crlf = buf.find("\r\n");
      std::string sizeLine = buf.substr(0, crlf);
      buf.erase(0, crlf + 2);
      if (auto semi = sizeLine.find(';'); semi != std::string::npos) {
        sizeLine.resize(semi);
      }
      size_t sz = 0;
      try {
        sz = std::stoul(sizeLine, nullptr, 16);
      } catch (...) {
        throw EtcdError("EtcdClient: bad chunk size");
      }
      if (sz == 0) break;
      while (buf.size() < sz + 2) readMore();
      bodyOut.append(buf, 0, sz);
      buf.erase(0, sz + 2);  // consume CRLF after chunk
    }
  } else if (findHeader(headers, "Content-Length", &contentLength)) {
    size_t want = 0;
    try {
      want = std::stoul(contentLength);
    } catch (...) {
      throw EtcdError("EtcdClient: bad Content-Length");
    }
    bodyOut = readExact(sock.getFd(), want, timeoutMs, already);
  } else {
    // Server-closes: read until EOF.
    bodyOut = std::move(already);
    while (true) {
      pollfd pfd{sock.getFd(), POLLIN, 0};
      int pr = ::poll(&pfd, 1, timeoutMs);
      if (pr <= 0) break;
      char tmp[4096];
      ssize_t r = ::recv(sock.getFd(), tmp, sizeof(tmp), 0);
      if (r <= 0) break;
      bodyOut.append(tmp, static_cast<size_t>(r));
    }
  }

  if (status / 100 != 2) {
    throw EtcdError(std::string("EtcdClient: HTTP ") + std::to_string(status) +
                    " from " + path + ": " + bodyOut);
  }
  return bodyOut;
}

Json::Value parseJson(const std::string& body) {
  Json::CharReaderBuilder b;
  Json::Value v;
  std::string errs;
  std::istringstream is(body);
  if (!Json::parseFromStream(b, is, &v, &errs)) {
    throw EtcdError("EtcdClient: response is not valid JSON: " + errs +
                    " body=" + body);
  }
  return v;
}

/// etcd JSON gateway returns numeric ids as strings ("ID":"123456"); accept
/// both for resilience.
int64_t toInt64(const Json::Value& v) {
  if (v.isInt64()) return v.asInt64();
  if (v.isUInt64()) return static_cast<int64_t>(v.asUInt64());
  if (v.isString()) {
    try {
      return std::stoll(v.asString());
    } catch (...) {
      throw EtcdError("EtcdClient: integer string not parseable: " +
                      v.asString());
    }
  }
  throw EtcdError("EtcdClient: expected integer, got " + v.toStyledString());
}

std::string serialize(const Json::Value& v) {
  Json::StreamWriterBuilder b;
  b["indentation"] = "";
  b["emitUTF8"] = true;
  return Json::writeString(b, v);
}

}  // namespace

EtcdClient::EtcdClient(const std::string& endpoint, int timeoutMs)
    : timeout_ms_(timeoutMs) {
  auto parsed = parseEtcdUrl(endpoint);
  host_ = parsed.host;
  port_ = parsed.port;
}

int64_t EtcdClient::leaseGrant(int64_t ttlSecs) {
  Json::Value body(Json::objectValue);
  body["TTL"] = static_cast<Json::Int64>(ttlSecs);
  body["ID"] = 0;
  auto resp = parseJson(httpPostJson(host_, port_, "/v3/lease/grant",
                                     serialize(body), timeout_ms_));
  if (!resp.isMember("ID")) {
    throw EtcdError("EtcdClient: lease/grant response missing ID: " +
                    resp.toStyledString());
  }
  return toInt64(resp["ID"]);
}

int64_t EtcdClient::leaseKeepAlive(int64_t leaseId) {
  Json::Value body(Json::objectValue);
  body["ID"] = static_cast<Json::Int64>(leaseId);
  auto resp = parseJson(httpPostJson(host_, port_, "/v3/lease/keepalive",
                                     serialize(body), timeout_ms_));
  // Streaming endpoint returns {"result": {"ID": "...", "TTL": "..."}}
  const auto& r = resp.isMember("result") ? resp["result"] : resp;
  if (!r.isMember("TTL")) return 0;
  return toInt64(r["TTL"]);
}

void EtcdClient::leaseRevoke(int64_t leaseId) {
  Json::Value body(Json::objectValue);
  body["ID"] = static_cast<Json::Int64>(leaseId);
  httpPostJson(host_, port_, "/v3/lease/revoke", serialize(body), timeout_ms_);
}

void EtcdClient::put(const std::string& key, const std::string& value,
                     int64_t leaseId) {
  Json::Value body(Json::objectValue);
  body["key"] = base64Encode(key);
  body["value"] = base64Encode(value);
  if (leaseId != 0) body["lease"] = static_cast<Json::Int64>(leaseId);
  httpPostJson(host_, port_, "/v3/kv/put", serialize(body), timeout_ms_);
}

void EtcdClient::deleteRange(const std::string& key) {
  Json::Value body(Json::objectValue);
  body["key"] = base64Encode(key);
  httpPostJson(host_, port_, "/v3/kv/deleterange", serialize(body),
               timeout_ms_);
}

}  // namespace tt::dynamo
