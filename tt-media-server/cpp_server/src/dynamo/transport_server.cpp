// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/transport_server.hpp"

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
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>
#include <utility>

#include "utils/logger.hpp"

namespace tt::dynamo {

namespace {

Json::Value parseJsonBytes(const uint8_t* data, size_t len) {
  Json::Value out;
  if (len == 0) return out;
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errs;
  const char* begin = reinterpret_cast<const char*>(data);
  if (!reader->parse(begin, begin + len, &out, &errs)) {
    return Json::Value{};
  }
  return out;
}

std::string dumpJsonCompact(const Json::Value& v) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["emitUTF8"] = true;
  return Json::writeString(writer, v);
}

std::string framedString(const TwoPartMessage& tp) {
  auto framed = encode_two_part(tp);
  return std::string(framed.begin(), framed.end());
}

}  // namespace

// ---------------------------------------------------------------------------
// DynamoTransportServer implementation
// ---------------------------------------------------------------------------

DynamoTransportServer::DynamoTransportServer(
    TransportServerConfig config, GenerateHandler handler,
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

DynamoTransportServer::~DynamoTransportServer() { shutdown(); }

void DynamoTransportServer::shutdown() {
  running_.store(false);
  if (tcp_server_) tcp_server_->stop();
}

void DynamoTransportServer::onMessage(const trantor::TcpConnectionPtr& conn,
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

void DynamoTransportServer::process_request(
    const trantor::TcpConnectionPtr& conn, const TcpRequestMessage& msg) {
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
          "[DynamoTransportServer] Request id={} input_tokens={} max_tokens={} "
          "address={}",
          id, genReq.token_ids.size(),
          genReq.max_tokens.has_value() ? std::to_string(*genReq.max_tokens)
                                        : "None",
          connInfo.address);
      handler_(genReq, connInfo);
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[DynamoTransportServer] request dispatch failed id={}: {}",
                   id, e.what());
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

void DynamoTransportServer::start() {
  const auto ioLoops =
      loop_pool_ ? loop_pool_->getLoops() : std::vector<trantor::EventLoop*>{};
  if (ioLoops.empty()) {
    throw std::runtime_error("DynamoTransportServer: no io loops provided");
  }

  running_.store(true);
  trantor::InetAddress addr(config_.bind_host, config_.bind_port);
  // Disable SO_REUSEPORT for fixed ports so a collision fails instead of
  // silently sharing the port; keep it for OS-assigned (port 0) binds.
  tcp_server_ = std::make_unique<trantor::TcpServer>(
      ioLoops.front(), addr, "DynamoTransportServer",
      /*reUseAddr=*/true, /*reUsePort=*/config_.bind_port == 0);
  tcp_server_->setIoLoops(ioLoops);
  // Port 0 is resolved during the acceptor's bind, which happens in the
  // TcpServer constructor — so the assigned port is available synchronously.
  actual_port_ = tcp_server_->address().toPort();

  tcp_server_->setAfterAcceptSockOptCallback([](int fd) {
    int one = 1;
    if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
      TT_LOG_WARN("[DynamoTransportServer] Failed to set TCP_NODELAY: {}",
                  strerror(errno));
    }
  });
  tcp_server_->setRecvMessageCallback(
      [this](const trantor::TcpConnectionPtr& conn, trantor::MsgBuffer* buf) {
        onMessage(conn, buf);
      });
  tcp_server_->start();

  TT_LOG_INFO(
      "[DynamoTransportServer] Listening on {}:{}  ({}.{}.{}, instance={})",
      config_.bind_host, actual_port_, config_.namespace_name,
      config_.component, config_.endpoint, config_.instance_id_hex);
}

}  // namespace tt::dynamo
