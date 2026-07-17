// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "dynamo/transport_protocol.hpp"

namespace trantor {
class EventLoop;
class EventLoopThreadPool;
class TcpServer;
class TcpClient;
class TcpConnection;
class MsgBuffer;
}  // namespace trantor

namespace tt::dynamo {

// ---------------------------------------------------------------------------
// Transport server
// ---------------------------------------------------------------------------

/**
 * Generate handler: invoked once per inbound request on the io loop. The
 * handler owns the call-home response stream (a DynamoStreamWriter) for the
 * supplied connection info and returns without blocking.
 */
using GenerateHandler = std::function<void(
    const GenerateRequest& request, const TcpStreamConnectionInfo& connInfo)>;

/**
 * Owns the outbound call-home connection for a single request and streams
 * framed response chunks to the frontend without blocking the caller.
 *
 * Mirrors the HTTP StreamingResponseWriter: token callbacks (on the LLMService
 * consumer thread) only `queueInLoop` onto the loop that owns the connection;
 * every socket write happens on that loop. A self-reference taken once the
 * connection is established keeps the writer (and its buffered sentinels) alive
 * until the peer closes, so a graceful shutdown is never truncated by the
 * TcpClient's abortive destructor.
 */
class DynamoStreamWriter
    : public std::enable_shared_from_this<DynamoStreamWriter> {
 public:
  static std::shared_ptr<DynamoStreamWriter> create(
      trantor::EventLoop* loop, TcpStreamConnectionInfo connInfo,
      std::string requestId, std::function<void()> onDisconnect);

  /// Dial the frontend's call-home address. Non-blocking.
  void connect();

  /// Queue a chunk for delivery. Returns false once the stream is terminal
  /// (error frame sent or peer gone), signalling the caller to stop.
  bool sendChunk(const TokenChunk& chunk);

  /// Queue the trailing sentinels and close. No-op once terminal.
  void finalize();

 private:
  DynamoStreamWriter(trantor::EventLoop* loop, TcpStreamConnectionInfo connInfo,
                     std::string requestId, std::function<void()> onDisconnect);

  void onConnState(const std::shared_ptr<trantor::TcpConnection>& conn);
  void ensurePrologue();
  void writeOrBuffer(std::string bytes);
  void closeStream();

  trantor::EventLoop* loop_;
  TcpStreamConnectionInfo conn_info_;
  std::string request_id_;
  std::function<void()> on_disconnect_;

  std::shared_ptr<trantor::TcpClient> client_;
  // The members below are touched only on `loop_`.
  std::shared_ptr<trantor::TcpConnection> conn_;
  std::shared_ptr<DynamoStreamWriter> self_guard_;
  std::vector<std::string> early_buffer_;
  bool connected_ = false;
  bool prologue_sent_ = false;
  bool closed_ = false;
  std::atomic<bool> done_{false};
};

struct TransportServerConfig {
  std::string bind_host = "0.0.0.0";
  uint16_t bind_port = 0;  // 0 = OS-assigned (recommended)
  std::string namespace_name = "default";
  std::string component = "backend";
  std::string endpoint = "generate";
  std::string model_name;
  std::string model_path;
  std::string instance_id_hex;  // Auto-generated when empty.
  uint64_t instance_id = 0;     // Auto-generated when zero.
};

class DynamoTransportServer {
 public:
  DynamoTransportServer(TransportServerConfig config, GenerateHandler handler,
                        trantor::EventLoopThreadPool* loopPool);
  ~DynamoTransportServer();

  DynamoTransportServer(const DynamoTransportServer&) = delete;
  DynamoTransportServer& operator=(const DynamoTransportServer&) = delete;

  /// Bind the listener on the supplied io loops and start serving.
  /// Non-blocking: the loops are driven by their own threads.
  void start();

  /// Stop the listener and force-close live connections. Safe to call from
  /// any thread.
  void shutdown();

  /// Bound port (valid after `start()`).
  uint16_t port() const { return actual_port_; }

  const TransportServerConfig& config() const { return config_; }

 private:
  TransportServerConfig config_;
  GenerateHandler handler_;
  trantor::EventLoopThreadPool* loop_pool_;
  std::unique_ptr<trantor::TcpServer> tcp_server_;
  uint16_t actual_port_ = 0;
  std::atomic<bool> running_{false};

  void onMessage(const std::shared_ptr<trantor::TcpConnection>& conn,
                 trantor::MsgBuffer* buf);
  void process_request(const std::shared_ptr<trantor::TcpConnection>& conn,
                       const TcpRequestMessage& msg);
};

}  // namespace tt::dynamo
