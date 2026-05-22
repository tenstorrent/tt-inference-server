// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_protocol.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <sstream>
#include <thread>
#include <utility>

#include "utils/logger.hpp"

namespace tt::dynamo {

namespace {

/// Parse JSON bytes into a Json::Value. Returns an empty value on parse error
/// (callers treat that as "skip").
Json::Value parseJsonBytes(const uint8_t* data, size_t len) {
  Json::Value outt;
  if (len == 0) return outt;
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errs;
  const char* begin = reinterpret_cast<const char*>(data);
  if (!reader->parse(begin, begin + len, &outt, &errs)) {
    return Json::Value{};
  }

  return outt;
}

std::string dumpJsonCompact(const Json::Value& v) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["emitUTF8"] = true;
  return Json::writeString(writer, v);
}

bool writeAll(int fd, const uint8_t* data, size_t len) {
  size_t written = 0;
  while (written < len) {
    ssize_t w = ::write(fd, data + written, len - written);
    if (w <= 0) return false;
    written += static_cast<size_t>(w);
  }
  return true;
}

bool writeAll(int fd, const std::vector<uint8_t>& data) {
  return writeAll(fd, data.data(), data.size());
}

}  // namespace

// ---------------------------------------------------------------------------
// Encoding / decoding
// ---------------------------------------------------------------------------

std::vector<uint8_t> encode_tcp_response(const std::vector<uint8_t>& data) {
  std::vector<uint8_t> buf;
  put_u32_be(buf, static_cast<uint32_t>(data.size()));
  buf.insert(buf.end(), data.begin(), data.end());
  return buf;
}

std::vector<uint8_t> encode_two_part(const TwoPartMessage& msg) {
  std::vector<uint8_t> buf;
  put_u64_be(buf, msg.header.size());
  put_u64_be(buf, msg.body.size());
  put_u64_be(buf, 0);  // checksum = 0 (release-mode)
  buf.insert(buf.end(), msg.header.begin(), msg.header.end());
  buf.insert(buf.end(), msg.body.begin(), msg.body.end());
  return buf;
}

TwoPartMessage decode_two_part(const std::vector<uint8_t>& data) {
  TwoPartMessage msg;
  if (data.size() < 24) return msg;

  uint64_t headerLen = get_u64_be(data.data());
  uint64_t bodyLen = get_u64_be(data.data() + 8);
  // Checksum at offset 16 is ignored.

  size_t headerStart = 24;
  size_t bodyStart = 24 + headerLen;
  if (data.size() < 24 + headerLen + bodyLen) return msg;

  msg.header.assign(data.begin() + headerStart,
                    data.begin() + headerStart + headerLen);
  msg.body.assign(data.begin() + bodyStart, data.begin() + bodyStart + bodyLen);
  return msg;
}

RequestControlMessage parse_control_message(
    const std::vector<uint8_t>& headerBytes) {
  RequestControlMessage ctrl;
  Json::Value j = parseJsonBytes(headerBytes.data(), headerBytes.size());
  if (!j.isObject()) return ctrl;

  ctrl.id = j.get("id", "").asString();
  ctrl.request_type = j.get("request_type", "").asString();
  ctrl.response_type = j.get("response_type", "").asString();

  if (j.isMember("connection_info")) {
    const Json::Value& ci = j["connection_info"];
    ctrl.connection_info.transport = ci.get("transport", "").asString();
    ctrl.connection_info.info = ci.get("info", "").asString();
  }
  return ctrl;
}

TcpStreamConnectionInfo parse_connection_info(const ConnectionInfo& info) {
  TcpStreamConnectionInfo tci;
  if (info.info.empty()) return tci;
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value j;
  std::string errs;
  if (!reader->parse(info.info.data(), info.info.data() + info.info.size(), &j,
                     &errs)) {
    return tci;
  }
  tci.address = j.get("address", "").asString();
  tci.subject = j.get("subject", "").asString();
  tci.context = j.get("context", "").asString();
  tci.stream_type = j.get("stream_type", "").asString();
  return tci;
}

GenerateRequest parse_generate_request(const std::vector<uint8_t>& bodyBytes) {
  GenerateRequest req;
  Json::Value j = parseJsonBytes(bodyBytes.data(), bodyBytes.size());
  if (!j.isObject()) return req;

  req.raw = j;
  req.model = j.get("model", "").asString();

  if (j.isMember("token_ids") && j["token_ids"].isArray()) {
    req.token_ids.reserve(j["token_ids"].size());
    for (const auto& t : j["token_ids"]) {
      req.token_ids.push_back(t.asInt());
    }
  }

  if (j.isMember("stop_conditions") && j["stop_conditions"].isObject()) {
    const auto& sc = j["stop_conditions"];
    if (sc.isMember("max_tokens") && !sc["max_tokens"].isNull()) {
      req.max_tokens = sc["max_tokens"].asInt();
    }
    if (sc.isMember("min_tokens") && !sc["min_tokens"].isNull()) {
      req.min_tokens = sc["min_tokens"].asInt();
    }
    if (sc.isMember("stop_token_ids") && sc["stop_token_ids"].isArray()) {
      for (const auto& t : sc["stop_token_ids"]) {
        req.stop_token_ids.push_back(t.asInt());
      }
    }
    if (sc.isMember("stop") && sc["stop"].isArray()) {
      for (const auto& s : sc["stop"]) {
        req.stop.push_back(s.asString());
      }
    }
    if (sc.isMember("ignore_eos") && sc["ignore_eos"].isBool()) {
      req.ignore_eos = sc["ignore_eos"].asBool();
    }
  }

  if (j.isMember("sampling_options") && j["sampling_options"].isObject()) {
    const auto& so = j["sampling_options"];
    auto getOptFloat = [&](const char* k) -> std::optional<float> {
      if (so.isMember(k) && !so[k].isNull()) return so[k].asFloat();
      return std::nullopt;
    };
    auto getOptInt = [&](const char* k) -> std::optional<int> {
      if (so.isMember(k) && !so[k].isNull()) return so[k].asInt();
      return std::nullopt;
    };
    req.temperature = getOptFloat("temperature");
    req.top_p = getOptFloat("top_p");
    req.top_k = getOptInt("top_k");
    req.seed = getOptInt("seed");
    req.frequency_penalty = getOptFloat("frequency_penalty");
    req.presence_penalty = getOptFloat("presence_penalty");
    req.repetition_penalty = getOptFloat("repetition_penalty");
  }

  return req;
}

std::vector<uint8_t> encode_stream_chunk(const TokenChunk& chunk) {
  Json::Value tokenData(Json::objectValue);
  Json::Value tokenIds(Json::arrayValue);
  for (int t : chunk.token_ids) tokenIds.append(t);
  tokenData["token_ids"] = std::move(tokenIds);
  if (chunk.finish_reason.has_value()) {
    tokenData["finish_reason"] = *chunk.finish_reason;
  } else {
    tokenData["finish_reason"] = Json::Value::null;
  }

  // Annotated<T>: {"data": <token_data>}
  Json::Value annotated(Json::objectValue);
  annotated["data"] = std::move(tokenData);

  // NetworkStreamWrapper: {"data": <annotated>, "complete_final": bool}
  Json::Value wrapper(Json::objectValue);
  wrapper["data"] = std::move(annotated);
  wrapper["complete_final"] = false;

  std::string s = dumpJsonCompact(wrapper);
  return std::vector<uint8_t>(s.begin(), s.end());
}

std::vector<uint8_t> encode_stream_final() {
  Json::Value wrapper(Json::objectValue);
  wrapper["data"] = Json::Value::null;
  wrapper["complete_final"] = true;
  std::string s = dumpJsonCompact(wrapper);
  return std::vector<uint8_t>(s.begin(), s.end());
}

// ---------------------------------------------------------------------------
// DynamoServer implementation
// ---------------------------------------------------------------------------

DynamoServer::DynamoServer(ServerConfig config, GenerateHandler handler)
    : config_(std::move(config)), handler_(std::move(handler)) {
  if (config_.instance_id == 0) {
    std::srand(static_cast<unsigned>(std::time(nullptr) ^ ::getpid()));
    config_.instance_id = (static_cast<uint64_t>(std::rand()) << 32) |
                          static_cast<uint64_t>(std::rand());
  }
  if (config_.instance_id_hex.empty()) {
    std::ostringstream oss;
    oss << std::hex << config_.instance_id;
    config_.instance_id_hex = oss.str();
  }
}

DynamoServer::~DynamoServer() { shutdown(); }

void DynamoServer::shutdown() {
  bool wasRunning = running_.exchange(false);
  if (listen_fd_ >= 0) {
    int fd = listen_fd_;
    listen_fd_ = -1;
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }
  (void)wasRunning;
}

bool DynamoServer::read_exact(int fd, std::vector<uint8_t>& buf, size_t n) {
  buf.resize(n);
  size_t total = 0;
  while (total < n) {
    ssize_t r = ::read(fd, buf.data() + total, n - total);
    if (r <= 0) return false;
    total += static_cast<size_t>(r);
  }
  return true;
}

bool DynamoServer::read_request(int fd, TcpRequestMessage& msg) {
  std::vector<uint8_t> tmp;
  if (!read_exact(fd, tmp, 2)) return false;
  uint16_t pathLen = get_u16_be(tmp.data());

  if (!read_exact(fd, tmp, pathLen)) return false;
  msg.endpoint_path.assign(tmp.begin(), tmp.end());

  if (!read_exact(fd, tmp, 2)) return false;
  uint16_t headersLen = get_u16_be(tmp.data());

  if (headersLen > 0) {
    if (!read_exact(fd, tmp, headersLen)) return false;
    Json::Value j = parseJsonBytes(tmp.data(), tmp.size());
    if (j.isObject()) {
      for (auto it = j.begin(); it != j.end(); ++it) {
        msg.headers[it.name()] = (*it).asString();
      }
    }
  }

  if (!read_exact(fd, tmp, 4)) return false;
  uint32_t payloadLen = get_u32_be(tmp.data());

  if (!read_exact(fd, tmp, payloadLen)) return false;
  msg.payload = std::move(tmp);
  return true;
}

void DynamoServer::process_request(int fd, const TcpRequestMessage& msg) {
  TwoPartMessage twoPart = decode_two_part(msg.payload);
  auto ctrl = parse_control_message(twoPart.header);
  auto genReq = parse_generate_request(twoPart.body);
  auto connInfo = parse_connection_info(ctrl.connection_info);

  TT_LOG_DEBUG(
      "[DynamoServer] Request id={} input_tokens={} max_tokens={} address={}",
      ctrl.id, genReq.token_ids.size(), genReq.max_tokens, connInfo.address);

  // ACK on the inbound connection.
  auto ack = encode_tcp_response();
  if (!writeAll(fd, ack)) {
    TT_LOG_WARN("[DynamoServer] Failed to send ACK for id={}", ctrl.id);
    return;
  }

  // Off-thread the slow path (LLMService dispatch + call-home streaming)
  // so the read loop on `fd` can immediately consume the next pipelined
  // request. `stream_response` opens its own outbound socket and never
  // touches `fd`, so concurrent streams don't share state.
  std::thread([this, connInfo = std::move(connInfo), requestId = ctrl.id,
               genReq = std::move(genReq)]() {
    stream_response(connInfo, requestId, genReq);
  }).detach();
}

void DynamoServer::stream_response(const TcpStreamConnectionInfo& connInfo,
                                   const std::string& requestId,
                                   const GenerateRequest& genReq) {
  auto colonPos = connInfo.address.rfind(':');
  if (colonPos == std::string::npos) {
    TT_LOG_ERROR("[DynamoServer] Invalid response address: {}",
                 connInfo.address);
    return;
  }
  std::string host = connInfo.address.substr(0, colonPos);
  uint16_t port =
      static_cast<uint16_t>(std::stoi(connInfo.address.substr(colonPos + 1)));

  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    TT_LOG_ERROR("[DynamoServer] Failed to create socket for response stream");
    return;
  }
  int flag = 1;
  ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

  if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) <
      0) {
    TT_LOG_ERROR("[DynamoServer] Failed to connect to response stream at {}",
                 connInfo.address);
    ::close(sock);
    return;
  }

  // 1. CallHomeHandshake (header-only TwoPartMessage).
  {
    Json::Value handshake(Json::objectValue);
    handshake["subject"] = connInfo.subject;
    handshake["stream_type"] = "response";
    std::string hs = dumpJsonCompact(handshake);

    TwoPartMessage tp;
    tp.header.assign(hs.begin(), hs.end());
    if (!writeAll(sock, encode_two_part(tp))) {
      TT_LOG_WARN("[DynamoServer] Failed to send handshake (id={})", requestId);
      ::close(sock);
      return;
    }
  }

  // 2. ResponseStreamPrologue (header-only).
  {
    Json::Value prologue(Json::objectValue);
    prologue["error"] = Json::Value::null;
    std::string ps = dumpJsonCompact(prologue);

    TwoPartMessage tp;
    tp.header.assign(ps.begin(), ps.end());
    if (!writeAll(sock, encode_two_part(tp))) {
      TT_LOG_WARN("[DynamoServer] Failed to send prologue (id={})", requestId);
      ::close(sock);
      return;
    }
  }

  // 3. User-supplied generate handler streams chunks back as data-only
  //    TwoPartMessages.
  handler_(genReq, [&](const TokenChunk& chunk) -> bool {
    auto chunkBytes = encode_stream_chunk(chunk);
    TwoPartMessage tp;
    tp.body = std::move(chunkBytes);
    return writeAll(sock, encode_two_part(tp));
  });

  // 4. complete_final sentinel.
  {
    auto finalBytes = encode_stream_final();
    TwoPartMessage tp;
    tp.body = std::move(finalBytes);
    writeAll(sock, encode_two_part(tp));
  }

  // 5. End-of-stream sentinel (empty TwoPartMessage).
  {
    TwoPartMessage tp;
    writeAll(sock, encode_two_part(tp));
  }

  ::close(sock);
}

void DynamoServer::handle_connection(int clientFd) {
  int flag = 1;
  ::setsockopt(clientFd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

  while (running_) {
    TcpRequestMessage msg;
    if (!read_request(clientFd, msg)) break;
    process_request(clientFd, msg);
  }

  ::close(clientFd);
}

void DynamoServer::run() {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    throw std::runtime_error("DynamoServer: failed to create listen socket");
  }

  int opt = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in bindAddr{};
  bindAddr.sin_family = AF_INET;
  bindAddr.sin_port = htons(config_.bind_port);
  ::inet_pton(AF_INET, config_.bind_host.c_str(), &bindAddr.sin_addr);

  if (::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&bindAddr),
             sizeof(bindAddr)) < 0) {
    throw std::runtime_error("DynamoServer: failed to bind");
  }

  struct sockaddr_in actual{};
  socklen_t len = sizeof(actual);
  ::getsockname(listen_fd_, reinterpret_cast<struct sockaddr*>(&actual), &len);
  actual_port_ = ntohs(actual.sin_port);

  if (::listen(listen_fd_, 128) < 0) {
    throw std::runtime_error("DynamoServer: failed to listen");
  }

  running_ = true;
  TT_LOG_INFO("[DynamoServer] Listening on {}:{}  ({}.{}.{}, instance={})",
              config_.bind_host, actual_port_, config_.namespace_name,
              config_.component, config_.endpoint, config_.instance_id_hex);

  while (running_) {
    int clientFd = ::accept(listen_fd_, nullptr, nullptr);
    if (clientFd < 0) {
      if (!running_) break;
      continue;
    }
    std::thread([this, clientFd]() { handle_connection(clientFd); }).detach();
  }
}

}  // namespace tt::dynamo
