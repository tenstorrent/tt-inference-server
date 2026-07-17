// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_protocol.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <trantor/net/EventLoop.h>
#include <trantor/net/EventLoopThreadPool.h>
#include <trantor/net/InetAddress.h>
#include <trantor/net/TcpClient.h>
#include <trantor/net/TcpConnection.h>
#include <trantor/net/TcpServer.h>
#include <trantor/utils/MsgBuffer.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string_view>
#include <utility>

#include "domain/llm/llm_error_reason.hpp"
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

std::string dumpJsonCompact(const Json::Value& v);  // defined below

/// Scan a flat JSON array of integers. `s[start]` must be '['. Appends each
/// value to `out` and returns the index just past the matching ']'. Returns
/// std::string_view::npos if the array is not a clean flat int array (nested
/// arrays, strings, floats, etc.) so the caller can fall back to the full JSON
/// parser. Tolerates whitespace, negative ints, and a trailing comma.
size_t scanIntArray(const char* s, size_t start, size_t n,
                    std::vector<uint32_t>& out) {
  if (start >= n || s[start] != '[') return std::string_view::npos;
  size_t i = start + 1;
  auto isWs = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  };
  while (i < n) {
    char c = s[i];
    if (isWs(c) || c == ',') {
      ++i;
      continue;
    }
    if (c == ']') return i + 1;
    bool neg = false;
    if (c == '-') {
      neg = true;
      ++i;
    }
    if (i >= n || s[i] < '0' || s[i] > '9') {
      return std::string_view::npos;  // not a plain integer element
    }
    long long v = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9') {
      v = v * 10 + (s[i] - '0');
      ++i;
    }
    out.push_back(static_cast<uint32_t>(neg ? -v : v));
  }
  return std::string_view::npos;  // unterminated array
}

/// Locate the object value for the top-level `"token_ids"` key and return the
/// index of its opening '['. Distinguishes the key from look-alikes such as
/// `"stop_token_ids_hidden"` by requiring the exact quoted key, immediately
/// preceded by '{' or ',' (an object key, not text inside a string value) and
/// followed by ':' then '['. Returns false if no such array is found.
bool findTokenIdsArray(std::string_view body, size_t& arrayStart) {
  static constexpr std::string_view kKey = "\"token_ids\"";
  auto isWs = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  };
  size_t pos = body.find(kKey);
  while (pos != std::string_view::npos) {
    // The char before the key (skipping whitespace) must open an object member.
    size_t b = pos;
    while (b > 0 && isWs(body[b - 1])) --b;
    const bool keyContext = b == 0 || body[b - 1] == '{' || body[b - 1] == ',';
    if (keyContext) {
      size_t i = pos + kKey.size();
      while (i < body.size() && isWs(body[i])) ++i;
      if (i < body.size() && body[i] == ':') {
        ++i;
        while (i < body.size() && isWs(body[i])) ++i;
        if (i < body.size() && body[i] == '[') {
          arrayStart = i;
          return true;
        }
      }
    }
    pos = body.find(kKey, pos + 1);
  }
  return false;
}

/// Populate every GenerateRequest field except `token_ids` from a parsed body.
/// Shared by the fast path (which parses a token_ids-elided copy) and the
/// jsoncpp fallback.
void populateGenerateFields(GenerateRequest& req, const Json::Value& j) {
  req.raw = j;
  req.model = j.get("model", "").asString();

  if (j.isMember("stop_conditions") && j["stop_conditions"].isObject()) {
    const auto& sc = j["stop_conditions"];
    if (sc.isMember("max_tokens") && !sc["max_tokens"].isNull()) {
      req.max_tokens = sc["max_tokens"].asInt();
    }
    if (sc.isMember("min_tokens") && !sc["min_tokens"].isNull()) {
      req.min_tokens = sc["min_tokens"].asInt();
    }
    // Dynamo's StopConditions serializes the token-id stop list as
    // `stop_token_ids_hidden` (see lib/llm/src/protocols/common.rs — the Rust
    // field has no serde rename). Prefer that key, but also accept the plain
    // `stop_token_ids` name for forward-compatibility / other senders.
    const Json::Value* stopIds = nullptr;
    if (sc.isMember("stop_token_ids_hidden") &&
        sc["stop_token_ids_hidden"].isArray()) {
      stopIds = &sc["stop_token_ids_hidden"];
    } else if (sc.isMember("stop_token_ids") &&
               sc["stop_token_ids"].isArray()) {
      stopIds = &sc["stop_token_ids"];
    }
    if (stopIds != nullptr) {
      for (const auto& t : *stopIds) {
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

  if (j.isMember("mm_processor_kwargs") && !j["mm_processor_kwargs"].isNull()) {
    req.mm_processor_kwargs = j["mm_processor_kwargs"];
    TT_LOG_INFO("[DynamoRx] mm_processor_kwargs received: {}",
                dumpJsonCompact(req.mm_processor_kwargs));
  }
}

std::string dumpJsonCompact(const Json::Value& v) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["emitUTF8"] = true;
  return Json::writeString(writer, v);
}

std::vector<uint8_t> encodeAnnotatedError(const std::string& message,
                                          uint16_t code) {
  // Encode the error message as a JSON payload with a status code so the
  // Dynamo frontend's extract_backend_error_if_present() can parse it and
  // return the correct HTTP status (e.g. 400 instead of default 500).
  Json::Value errorPayload(Json::objectValue);
  errorPayload["message"] = message;
  errorPayload["code"] = code;
  std::string errorJson = dumpJsonCompact(errorPayload);

  Json::Value annotated(Json::objectValue);
  annotated["data"] = Json::Value::null;
  annotated["event"] = "error";
  Json::Value comment(Json::arrayValue);
  comment.append(errorJson);
  annotated["comment"] = std::move(comment);

  Json::Value wrapper(Json::objectValue);
  wrapper["data"] = std::move(annotated);
  wrapper["complete_final"] = false;

  std::string s = dumpJsonCompact(wrapper);
  return std::vector<uint8_t>(s.begin(), s.end());
}

std::string framedString(const TwoPartMessage& tp) {
  auto framed = encode_two_part(tp);
  return std::string(framed.begin(), framed.end());
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
  if (bodyBytes.empty()) return req;
  const char* data = reinterpret_cast<const char*>(bodyBytes.data());
  const size_t n = bodyBytes.size();
  const std::string_view body(data, n);

  size_t arrayStart = 0;
  if (findTokenIdsArray(body, arrayStart)) {
    std::vector<uint32_t> ids;
    const size_t arrayEnd = scanIntArray(data, arrayStart, n, ids);
    if (arrayEnd != std::string_view::npos) {
      std::string reduced;
      reduced.reserve(arrayStart + 2 + (n - arrayEnd));
      reduced.append(data, arrayStart);
      reduced.append("[]");
      reduced.append(data + arrayEnd, n - arrayEnd);
      Json::Value j = parseJsonBytes(
          reinterpret_cast<const uint8_t*>(reduced.data()), reduced.size());
      if (j.isObject()) {
        req.token_ids = std::move(ids);
        populateGenerateFields(req, j);
        return req;
      }
    }
  }

  // Fallback: full jsoncpp parse (handles any body the fast path declined).
  Json::Value j = parseJsonBytes(bodyBytes.data(), n);
  if (!j.isObject()) return req;
  if (j.isMember("token_ids") && j["token_ids"].isArray()) {
    req.token_ids.reserve(j["token_ids"].size());
    for (const auto& t : j["token_ids"]) {
      req.token_ids.push_back(static_cast<uint32_t>(t.asUInt()));
    }
  }
  populateGenerateFields(req, j);
  return req;
}

std::vector<uint8_t> encode_stream_chunk(const TokenChunk& chunk) {
  // If this chunk carries an error, encode as Annotated::error format
  // so the Dynamo frontend's check_for_backend_error can intercept it.
  if (chunk.error.has_value()) {
    return encodeAnnotatedError(*chunk.error, chunk.error_code.value_or(500));
  }

  if (chunk.finish_reason.has_value() &&
      tt::domain::llm::isErrorFinishReason(*chunk.finish_reason)) {
    return encodeAnnotatedError(*chunk.finish_reason, 500);
  }

  Json::Value tokenData(Json::objectValue);
  Json::Value tokenIds(Json::arrayValue);
  for (uint32_t t : chunk.token_ids) tokenIds.append(t);
  tokenData["token_ids"] = std::move(tokenIds);
  if (chunk.finish_reason.has_value()) {
    tokenData["finish_reason"] = *chunk.finish_reason;
  } else {
    tokenData["finish_reason"] = Json::Value::null;
  }

  // BackendOutput.completion_usage (async-openai CompletionUsage shape). The
  // frontend's chat/completions aggregator copies prompt_tokens_details from
  // here; the (currently missing) completion_tokens_details copy is what makes
  // reasoning_tokens surface once the frontend is patched.
  if (chunk.completion_usage.has_value()) {
    const auto& u = *chunk.completion_usage;
    Json::Value cu(Json::objectValue);
    cu["prompt_tokens"] = u.prompt_tokens;
    cu["completion_tokens"] = u.completion_tokens;
    cu["total_tokens"] = u.total_tokens;
    if (u.cached_tokens.has_value()) {
      Json::Value ptd(Json::objectValue);
      ptd["cached_tokens"] = *u.cached_tokens;
      cu["prompt_tokens_details"] = std::move(ptd);
    }
    if (u.reasoning_tokens.has_value()) {
      Json::Value ctd(Json::objectValue);
      ctd["reasoning_tokens"] = *u.reasoning_tokens;
      cu["completion_tokens_details"] = std::move(ctd);
    }
    tokenData["completion_usage"] = std::move(cu);
  }

  if (!chunk.engine_data.isNull()) {
    tokenData["engine_data"] = chunk.engine_data;
  }
  if (!chunk.disaggregated_params.isNull()) {
    tokenData["disaggregated_params"] = chunk.disaggregated_params;
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

DynamoServer::DynamoServer(ServerConfig config, GenerateHandler handler,
                           trantor::EventLoopThreadPool* loopPool)
    : config_(std::move(config)),
      handler_(std::move(handler)),
      loop_pool_(loopPool) {
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
  running_.store(false);
  if (tcp_server_) tcp_server_->stop();
}

void DynamoServer::onMessage(const trantor::TcpConnectionPtr& conn,
                             trantor::MsgBuffer* buf) {
  // Frame: [path_len:u16][path][headers_len:u16][headers][payload_len:u32]
  //        [payload]. Extract every complete frame the buffer holds; leave a
  // partial frame for the next callback.
  while (running_.load()) {
    const size_t avail = buf->readableBytes();
    if (avail < 2) return;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf->peek());

    const uint16_t pathLen = get_u16_be(p);
    const size_t afterPath = 2u + pathLen + 2u;
    if (avail < afterPath) return;

    const uint16_t headersLen = get_u16_be(p + 2 + pathLen);
    const size_t afterHeaders = afterPath + headersLen + 4u;
    if (avail < afterHeaders) return;

    const uint32_t payloadLen = get_u32_be(p + afterPath + headersLen);
    const size_t total = afterHeaders + payloadLen;
    if (avail < total) return;

    TcpRequestMessage msg;
    msg.endpoint_path.assign(p + 2, p + 2 + pathLen);
    if (headersLen > 0) {
      Json::Value j = parseJsonBytes(p + afterPath, headersLen);
      if (j.isObject()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
          msg.headers[it.name()] = (*it).asString();
        }
      }
    }
    msg.payload.assign(p + afterHeaders, p + total);
    buf->retrieve(total);

    process_request(conn, msg);
  }
}

void DynamoServer::process_request(const trantor::TcpConnectionPtr& conn,
                                   const TcpRequestMessage& msg) {
  TwoPartMessage twoPart = decode_two_part(msg.payload);
  auto ctrl = parse_control_message(twoPart.header);
  auto connInfo = parse_connection_info(ctrl.connection_info);

  auto ack = encode_tcp_response();
  conn->send(reinterpret_cast<const char*>(ack.data()), ack.size());

  trantor::EventLoop* loop = loop_pool_->getNextLoop();
  loop->queueInLoop([this, body = std::move(twoPart.body),
                     connInfo = std::move(connInfo), id = ctrl.id]() mutable {
    try {
      GenerateRequest genReq = parse_generate_request(body);
      TT_LOG_DEBUG(
          "[DynamoServer] Request id={} input_tokens={} max_tokens={} "
          "address={}",
          id, genReq.token_ids.size(),
          genReq.max_tokens.has_value() ? std::to_string(*genReq.max_tokens)
                                        : "None",
          connInfo.address);
      handler_(genReq, connInfo);
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[DynamoServer] request dispatch failed id={}: {}", id,
                   e.what());
    }
  });
}

// ---------------------------------------------------------------------------
// DynamoStreamWriter
// ---------------------------------------------------------------------------

std::shared_ptr<DynamoStreamWriter> DynamoStreamWriter::create(
    trantor::EventLoop* loop, TcpStreamConnectionInfo connInfo,
    std::string requestId, std::function<void()> onDisconnect) {
  return std::shared_ptr<DynamoStreamWriter>(
      new DynamoStreamWriter(loop, std::move(connInfo), std::move(requestId),
                             std::move(onDisconnect)));
}

DynamoStreamWriter::DynamoStreamWriter(trantor::EventLoop* loop,
                                       TcpStreamConnectionInfo connInfo,
                                       std::string requestId,
                                       std::function<void()> onDisconnect)
    : loop_(loop),
      conn_info_(std::move(connInfo)),
      request_id_(std::move(requestId)),
      on_disconnect_(std::move(onDisconnect)) {}

void DynamoStreamWriter::connect() {
  const auto colon = conn_info_.address.rfind(':');
  uint16_t port = 0;
  bool ok = colon != std::string::npos;
  std::string host;
  if (ok) {
    host = conn_info_.address.substr(0, colon);
    try {
      port = static_cast<uint16_t>(
          std::stoi(conn_info_.address.substr(colon + 1)));
    } catch (const std::exception&) {
      ok = false;
    }
  }
  if (!ok) {
    TT_LOG_ERROR("[DynamoTx] id={} invalid response address: {}", request_id_,
                 conn_info_.address);
    if (!done_.exchange(true) && on_disconnect_) on_disconnect_();
    return;
  }

  client_ = std::make_shared<trantor::TcpClient>(
      loop_, trantor::InetAddress(host, port), "DynamoCallHome");
  std::weak_ptr<DynamoStreamWriter> weak = shared_from_this();
  client_->setConnectionCallback([weak](const trantor::TcpConnectionPtr& c) {
    if (auto self = weak.lock()) self->onConnState(c);
  });
  client_->setConnectionErrorCallback([weak]() {
    if (auto self = weak.lock()) {
      TT_LOG_WARN("[DynamoTx] id={} call-home connect failed",
                  self->request_id_);
      if (!self->done_.exchange(true) && self->on_disconnect_)
        self->on_disconnect_();
    }
  });
  client_->connect();
}

void DynamoStreamWriter::onConnState(const trantor::TcpConnectionPtr& conn) {
  if (conn->connected()) {
    conn->setTcpNoDelay(true);
    conn_ = conn;
    connected_ = true;
    // Stay alive until the peer closes so a graceful shutdown is not truncated
    // by ~TcpClient's forceClose once the pipeline callbacks are released.
    self_guard_ = shared_from_this();

    Json::Value handshake(Json::objectValue);
    handshake["subject"] = conn_info_.subject;
    handshake["stream_type"] = "response";
    std::string hs = dumpJsonCompact(handshake);
    TwoPartMessage tp;
    tp.header.assign(hs.begin(), hs.end());
    conn_->send(framedString(tp));

    for (const auto& b : early_buffer_) conn_->send(b);
    early_buffer_.clear();
    if (closed_) conn_->shutdown();
  } else {
    connected_ = false;
    conn_.reset();
    if (!done_.exchange(true) && on_disconnect_) on_disconnect_();
    // Release the self-reference on a fresh tick so we never destroy the
    // TcpClient from inside its own callback.
    if (self_guard_) loop_->queueInLoop([g = std::move(self_guard_)]() {});
  }
}

void DynamoStreamWriter::ensurePrologue() {
  if (prologue_sent_) return;
  prologue_sent_ = true;
  Json::Value prologue(Json::objectValue);
  prologue["error"] = Json::Value::null;
  std::string ps = dumpJsonCompact(prologue);
  TwoPartMessage tp;
  tp.header.assign(ps.begin(), ps.end());
  writeOrBuffer(framedString(tp));
}

void DynamoStreamWriter::writeOrBuffer(std::string bytes) {
  if (connected_ && conn_)
    conn_->send(bytes);
  else
    early_buffer_.push_back(std::move(bytes));
}

void DynamoStreamWriter::closeStream() {
  if (closed_) return;
  closed_ = true;
  if (connected_ && conn_) conn_->shutdown();
}

bool DynamoStreamWriter::sendChunk(const TokenChunk& chunk) {
  if (done_.load()) return false;
  const bool isError = chunk.error.has_value();

  TwoPartMessage tp;
  auto body = encode_stream_chunk(chunk);
  tp.body.assign(body.begin(), body.end());
  std::string bytes = framedString(tp);

  if (isError) done_.store(true);
  auto self = shared_from_this();
  loop_->queueInLoop([self, bytes = std::move(bytes), isError]() mutable {
    if (self->closed_) return;
    self->ensurePrologue();
    self->writeOrBuffer(std::move(bytes));
    if (isError) self->closeStream();  // error frames carry no sentinels
  });
  return !isError;
}

void DynamoStreamWriter::finalize() {
  done_.store(true);
  auto self = shared_from_this();
  loop_->queueInLoop([self]() {
    if (self->closed_) return;
    self->ensurePrologue();
    {
      TwoPartMessage tp;
      auto f = encode_stream_final();
      tp.body.assign(f.begin(), f.end());
      self->writeOrBuffer(framedString(tp));
    }
    self->writeOrBuffer(framedString(TwoPartMessage{}));  // end-of-stream
    self->closeStream();
  });
}

void DynamoServer::start() {
  const auto ioLoops =
      loop_pool_ ? loop_pool_->getLoops() : std::vector<trantor::EventLoop*>{};
  if (ioLoops.empty()) {
    throw std::runtime_error("DynamoServer: no io loops provided");
  }

  running_.store(true);
  trantor::InetAddress addr(config_.bind_host, config_.bind_port);
  // Disable SO_REUSEPORT for fixed ports so a collision fails instead of
  // silently sharing the port; keep it for OS-assigned (port 0) binds.
  tcp_server_ = std::make_unique<trantor::TcpServer>(
      ioLoops.front(), addr, "DynamoServer",
      /*reUseAddr=*/true, /*reUsePort=*/config_.bind_port == 0);
  tcp_server_->setIoLoops(ioLoops);
  // Port 0 is resolved during the acceptor's bind, which happens in the
  // TcpServer constructor — so the assigned port is available synchronously.
  actual_port_ = tcp_server_->address().toPort();

  tcp_server_->setAfterAcceptSockOptCallback([](int fd) {
    int one = 1;
    if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
      TT_LOG_WARN("[DynamoServer] Failed to set TCP_NODELAY: {}",
                  strerror(errno));
    }
  });
  tcp_server_->setRecvMessageCallback(
      [this](const trantor::TcpConnectionPtr& conn, trantor::MsgBuffer* buf) {
        onMessage(conn, buf);
      });
  tcp_server_->start();

  TT_LOG_INFO("[DynamoServer] Listening on {}:{}  ({}.{}.{}, instance={})",
              config_.bind_host, actual_port_, config_.namespace_name,
              config_.component, config_.endpoint, config_.instance_id_hex);
}

}  // namespace tt::dynamo
