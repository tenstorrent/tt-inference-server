// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_protocol.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <trantor/net/EventLoop.h>
#include <trantor/net/InetAddress.h>
#include <trantor/net/TcpConnection.h>
#include <trantor/net/TcpServer.h>
#include <trantor/utils/MsgBuffer.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>
#include <utility>

#include "utils/logger.hpp"

namespace tt::dynamo {

namespace {

/// Enables verbose per-chunk send tracing. Set DYN_TX_TRACE=1 (or any
/// non-zero/non-"false" value) to log a [DynamoTx] line for every
/// TokenChunk written to the call-home socket. Useful for verifying
/// whether inter-token latency is incurred on the wire (here) or in the
/// frontend's SSE delivery path.
///
/// Read once on first call; result is cached for the lifetime of the
/// process so we don't pay getenv() per token.
bool txTraceEnabled() {
  static const bool kEnabled = [] {
    const char* v = std::getenv("DYN_TX_TRACE");
    if (v == nullptr) return false;
    std::string s(v);
    if (s.empty() || s == "0" || s == "false" || s == "FALSE" || s == "False" ||
        s == "off" || s == "OFF") {
      return false;
    }
    return true;
  }();
  return kEnabled;
}

using SteadyClock = std::chrono::steady_clock;

int64_t microsBetween(SteadyClock::time_point a, SteadyClock::time_point b) {
  return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

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

  return req;
}

std::vector<uint8_t> encode_stream_chunk(const TokenChunk& chunk) {
  // If this chunk carries an error, encode as Annotated::error format
  // so the Dynamo frontend's check_for_backend_error can intercept it.
  if (chunk.error.has_value()) {
    // Encode the error message as a JSON payload with a status code so the
    // Dynamo frontend's extract_backend_error_if_present() can parse it and
    // return the correct HTTP status (e.g. 400 instead of default 500).
    Json::Value errorPayload(Json::objectValue);
    errorPayload["message"] = *chunk.error;
    errorPayload["code"] = chunk.error_code.value_or(500);
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

  Json::Value tokenData(Json::objectValue);
  Json::Value tokenIds(Json::arrayValue);
  for (int t : chunk.token_ids) tokenIds.append(t);
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
                           std::vector<trantor::EventLoop*> io_loops)
    : config_(std::move(config)),
      handler_(std::move(handler)),
      io_loops_(std::move(io_loops)) {
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
  auto genReq = parse_generate_request(twoPart.body);
  auto connInfo = parse_connection_info(ctrl.connection_info);

  TT_LOG_DEBUG(
      "[DynamoServer] Request id={} input_tokens={} max_tokens={} address={}",
      ctrl.id, genReq.token_ids.size(), genReq.max_tokens, connInfo.address);

  // ACK on the inbound connection. onMessage runs on the connection's io
  // loop, so sends stay in the FIFO order the frontend pipelined requests in.
  auto ack = encode_tcp_response();
  conn->send(reinterpret_cast<const char*>(ack.data()), ack.size());

  // Off-thread the slow path (LLMService dispatch + call-home streaming) so
  // the io loop can keep reading the next pipelined request. `stream_response`
  // opens its own outbound socket and never touches the inbound connection.
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

  const bool txTrace = txTraceEnabled();

  // Timing reference for the whole call-home leg. Stage timings are deltas
  // from this point so we can correlate against the frontend's
  // worker_recv_to_first_chunk_ms / dispatch_to_first_chunk_ms logs and
  // against client-side TTFT/ITL measurements.
  const auto tStart = SteadyClock::now();

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

  const auto tConnectStart = SteadyClock::now();
  if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) <
      0) {
    TT_LOG_ERROR("[DynamoServer] Failed to connect to response stream at {}",
                 connInfo.address);
    ::close(sock);
    return;
  }
  const auto tConnected = SteadyClock::now();
  TT_LOG_INFO("[DynamoTx] id={} stage=connected connect_us={}", requestId,
              microsBetween(tConnectStart, tConnected));

  // 1. CallHomeHandshake (header-only TwoPartMessage).
  {
    Json::Value handshake(Json::objectValue);
    handshake["subject"] = connInfo.subject;
    handshake["stream_type"] = "response";
    std::string hs = dumpJsonCompact(handshake);

    TwoPartMessage tp;
    tp.header.assign(hs.begin(), hs.end());
    const auto tBefore = SteadyClock::now();
    if (!writeAll(sock, encode_two_part(tp))) {
      TT_LOG_WARN("[DynamoServer] Failed to send handshake (id={})", requestId);
      ::close(sock);
      return;
    }
    const auto tAfter = SteadyClock::now();
    TT_LOG_INFO(
        "[DynamoTx] id={} stage=handshake_sent write_us={} since_start_us={}",
        requestId, microsBetween(tBefore, tAfter),
        microsBetween(tStart, tAfter));
  }

  // 2. ResponseStreamPrologue + streaming.
  //    The prologue carries either error=null (success) or error="<msg>"
  //    (pre-stream rejection). We defer sending the prologue until the first
  //    chunk arrives so the handler can signal an error via TokenChunk::error.
  size_t chunkSeq = 0;
  size_t totalTokens = 0;
  size_t totalBytes = 0;
  SteadyClock::time_point tFirstChunk{};
  SteadyClock::time_point tPrevChunk{};
  bool sawFirstChunk = false;
  bool prologueSent = false;
  bool hadError = false;

  auto sendPrologue = [&](const Json::Value& errorVal) -> bool {
    Json::Value prologue(Json::objectValue);
    prologue["error"] = errorVal;
    std::string ps = dumpJsonCompact(prologue);

    TwoPartMessage tp;
    tp.header.assign(ps.begin(), ps.end());
    const auto tBefore = SteadyClock::now();
    if (!writeAll(sock, encode_two_part(tp))) {
      TT_LOG_WARN("[DynamoServer] Failed to send prologue (id={})", requestId);
      return false;
    }
    const auto tAfter = SteadyClock::now();
    TT_LOG_INFO(
        "[DynamoTx] id={} stage=prologue_sent write_us={} since_start_us={}",
        requestId, microsBetween(tBefore, tAfter),
        microsBetween(tStart, tAfter));
    prologueSent = true;
    return true;
  };

  handler_(genReq, [&](const TokenChunk& chunk) -> bool {
    // Always send a success prologue first (even for error chunks).
    if (!prologueSent) {
      if (!sendPrologue(Json::Value::null)) {
        return false;
      }
    }

    // If the chunk carries an error, send it as an Annotated::error frame and
    // abort. The Dynamo frontend's check_for_backend_error() will intercept
    // this and map it to an HTTP 4xx/5xx response.
    if (chunk.error.has_value()) {
      hadError = true;
      // Send the error as an Annotated::error frame, then stop.
      auto chunkBytes = encode_stream_chunk(chunk);
      TwoPartMessage errTp;
      errTp.body = std::move(chunkBytes);
      writeAll(sock, encode_two_part(errTp));
      return false;
    }

    auto chunkBytes = encode_stream_chunk(chunk);
    const size_t bytesOut = chunkBytes.size() + sizeof(uint64_t) * 3;
    TwoPartMessage tp;
    tp.body = std::move(chunkBytes);
    auto framed = encode_two_part(tp);

    const auto tBefore = SteadyClock::now();
    const bool ok = writeAll(sock, framed);
    const auto tAfter = SteadyClock::now();

    if (!ok) {
      TT_LOG_WARN(
          "[DynamoTx] id={} stage=chunk_failed seq={} tokens={} bytes={}",
          requestId, chunkSeq, chunk.token_ids.size(), bytesOut);
      return false;
    }

    if (!sawFirstChunk) {
      tFirstChunk = tBefore;
      tPrevChunk = tBefore;
      sawFirstChunk = true;
    }

    if (txTrace) {
      // Per-chunk INFO line. High volume (~one per generated token) — only
      // emitted when DYN_TX_TRACE is set. Format is intentionally compact
      // and easy to grep/awk for offline analysis.
      //   seq            — chunk index within this request
      //   tokens         — token ids packed into this chunk (usually 1)
      //   bytes          — framed wire bytes (TwoPartMessage envelope incl.)
      //   write_us       — time spent inside writeAll() (kernel buffer copy)
      //   since_prev_us  — gap since the previous chunk's write(); this is
      //                    the per-token inter-arrival the wire sees
      //   since_first_us — offset from the first chunk; matches client TTFT
      //                    + cumulative ITL
      TT_LOG_INFO(
          "[DynamoTx] id={} stage=chunk seq={} tokens={} bytes={} "
          "write_us={} since_prev_us={} since_first_us={}",
          requestId, chunkSeq, chunk.token_ids.size(), bytesOut,
          microsBetween(tBefore, tAfter), microsBetween(tPrevChunk, tBefore),
          microsBetween(tFirstChunk, tBefore));
    }

    tPrevChunk = tBefore;
    chunkSeq++;
    totalTokens += chunk.token_ids.size();
    totalBytes += bytesOut;
    return true;
  });

  // If an error was sent in the prologue, skip the stream sentinels.
  if (hadError) {
    ::close(sock);
    return;
  }

  // Send prologue now if the handler produced no chunks at all.
  if (!prologueSent) {
    if (!sendPrologue(Json::Value::null)) {
      ::close(sock);
      return;
    }
  }

  // 3. complete_final sentinel.
  {
    auto finalBytes = encode_stream_final();
    TwoPartMessage tp;
    tp.body = std::move(finalBytes);
    const auto tBefore = SteadyClock::now();
    writeAll(sock, encode_two_part(tp));
    const auto tAfter = SteadyClock::now();
    TT_LOG_INFO(
        "[DynamoTx] id={} stage=complete_final chunks={} tokens={} bytes={} "
        "stream_us={} write_us={}",
        requestId, chunkSeq, totalTokens, totalBytes,
        sawFirstChunk ? microsBetween(tFirstChunk, tBefore) : 0,
        microsBetween(tBefore, tAfter));
  }

  // 4. End-of-stream sentinel (empty TwoPartMessage).
  {
    TwoPartMessage tp;
    writeAll(sock, encode_two_part(tp));
  }

  ::close(sock);
}

void DynamoServer::start() {
  if (io_loops_.empty()) {
    throw std::runtime_error("DynamoServer: no io loops provided");
  }

  running_.store(true);
  trantor::InetAddress addr(config_.bind_host, config_.bind_port);
  tcp_server_ = std::make_unique<trantor::TcpServer>(io_loops_.front(), addr,
                                                     "DynamoServer");
  tcp_server_->setIoLoops(io_loops_);
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
