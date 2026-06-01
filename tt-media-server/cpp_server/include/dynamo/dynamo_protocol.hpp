// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * Dynamo TCP Wire Protocol
 *
 * Implements NVIDIA Dynamo's request plane so this C++ server can register
 * as a native backend worker without a Python bridge. Originally adapted
 * from https://github.com/ai-dynamo/dynamo and the dynamo-mock-backend
 * sibling repo.
 *
 * Wire formats implemented:
 *   - TCP request frame (ingress from frontend)
 *   - TCP response frame (ACK back to frontend)
 *   - TwoPartCodec (header+body framing for control + data)
 *   - Call-home response stream (streaming tokens back to the frontend)
 */

#include <json/json.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::dynamo {

// ---------------------------------------------------------------------------
// Wire-format helpers
// ---------------------------------------------------------------------------

inline void put_u16_be(std::vector<uint8_t>& buf, uint16_t val) {
  buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
  buf.push_back(static_cast<uint8_t>(val & 0xFF));
}

inline void put_u32_be(std::vector<uint8_t>& buf, uint32_t val) {
  buf.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
  buf.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
  buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
  buf.push_back(static_cast<uint8_t>(val & 0xFF));
}

inline void put_u64_be(std::vector<uint8_t>& buf, uint64_t val) {
  for (int i = 7; i >= 0; --i) {
    buf.push_back(static_cast<uint8_t>((val >> (i * 8)) & 0xFF));
  }
}

inline uint16_t get_u16_be(const uint8_t* p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) |
                               static_cast<uint16_t>(p[1]));
}

inline uint32_t get_u32_be(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
         (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

inline uint64_t get_u64_be(const uint8_t* p) {
  uint64_t val = 0;
  for (int i = 0; i < 8; ++i) {
    val = (val << 8) | static_cast<uint64_t>(p[i]);
  }
  return val;
}

// ---------------------------------------------------------------------------
// TCP request frame
// Wire: [path_len:u16][path][headers_len:u16][headers_json][payload_len:u32]
//       [payload]
// ---------------------------------------------------------------------------

struct TcpRequestMessage {
  std::string endpoint_path;
  std::unordered_map<std::string, std::string> headers;
  std::vector<uint8_t> payload;
};

// ---------------------------------------------------------------------------
// TCP response frame
// Wire: [length:u32][data]   (empty data = success ACK)
// ---------------------------------------------------------------------------

std::vector<uint8_t> encode_tcp_response(const std::vector<uint8_t>& data = {});

// ---------------------------------------------------------------------------
// TwoPartCodec
// Wire: [header_len:u64][body_len:u64][checksum:u64][header][body]
// ---------------------------------------------------------------------------

struct TwoPartMessage {
  std::vector<uint8_t> header;
  std::vector<uint8_t> body;
};

std::vector<uint8_t> encode_two_part(const TwoPartMessage& msg);
TwoPartMessage decode_two_part(const std::vector<uint8_t>& data);

// ---------------------------------------------------------------------------
// RequestControlMessage (header of the inbound TwoPartMessage)
//
// JSON shape: {"id": str, "request_type": "single_in",
//              "response_type": "many_out",
//              "connection_info": {"transport": "tcp_server",
//                                  "info": "<inner JSON string>"}}
// ---------------------------------------------------------------------------

struct ConnectionInfo {
  std::string transport;  // e.g. "tcp_server"
  std::string info;       // JSON string: {"address","subject","context",...}
};

struct RequestControlMessage {
  std::string id;
  std::string request_type;
  std::string response_type;
  ConnectionInfo connection_info;
};

RequestControlMessage parse_control_message(
    const std::vector<uint8_t>& header_bytes);

struct TcpStreamConnectionInfo {
  std::string address;      // "host:port"
  std::string subject;      // UUID
  std::string context;      // request id
  std::string stream_type;  // e.g. "Response"
};

TcpStreamConnectionInfo parse_connection_info(const ConnectionInfo& info);

// ---------------------------------------------------------------------------
// Streaming response chunk produced by the generate handler.
// ---------------------------------------------------------------------------

/// Optional token-accounting the worker reports to the frontend on the final
/// chunk. Serialized as the OpenAI/async-openai `CompletionUsage` shape, which
/// the Dynamo frontend reads from `BackendOutput.completion_usage` to populate
/// the response `usage` block (prompt_tokens_details / completion_tokens_details).
struct DynamoUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  /// -> usage.prompt_tokens_details.cached_tokens (prefix-cache reuse)
  std::optional<int> cached_tokens;
  /// -> usage.completion_tokens_details.reasoning_tokens (<think>…</think> span)
  std::optional<int> reasoning_tokens;
};

struct TokenChunk {
  std::vector<int> token_ids;
  std::optional<std::string> finish_reason;
  /// If set, signals a pre-stream error. stream_response will send this as an
  /// Annotated::error chunk so the Dynamo frontend can intercept it.
  std::optional<std::string> error;
  /// HTTP status code to return when error is set (default: 500).
  std::optional<uint16_t> error_code;
  /// Populated on the final chunk only; serialized as `completion_usage`.
  std::optional<DynamoUsage> completion_usage;
};

/// Encode a TokenChunk as a NetworkStreamWrapper<Annotated<T>> JSON body.
std::vector<uint8_t> encode_stream_chunk(const TokenChunk& chunk);

/// Encode the trailing `complete_final` sentinel.
std::vector<uint8_t> encode_stream_final();

// ---------------------------------------------------------------------------
// Generate request body (body of the inbound TwoPartMessage). Mirrors
// dynamo's `PreprocessedRequest` minimally; full JSON kept in `raw` for
// forward compatibility.
// ---------------------------------------------------------------------------

struct GenerateRequest {
  std::string model;
  std::vector<int> token_ids;

  // StopConditions ---------------------------------------------------------
  int max_tokens = 128;
  std::optional<int> min_tokens;
  std::vector<int> stop_token_ids;
  std::vector<std::string> stop;
  bool ignore_eos = false;

  // SamplingOptions --------------------------------------------------------
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> top_k;
  std::optional<int> seed;
  std::optional<float> frequency_penalty;
  std::optional<float> presence_penalty;
  std::optional<float> repetition_penalty;

  Json::Value raw;  // Full parsed JSON body, for any field we don't yet map.
};

GenerateRequest parse_generate_request(const std::vector<uint8_t>& body_bytes);

// ---------------------------------------------------------------------------
// High-level server
// ---------------------------------------------------------------------------

/**
 * Generate handler: invoked once per inbound request. Push tokens via
 * `send_chunk`; the server framework calls the call-home response stream and
 * the trailing sentinels for you.
 */
using GenerateHandler =
    std::function<void(const GenerateRequest& request,
                       std::function<bool(const TokenChunk&)> send_chunk)>;

struct ServerConfig {
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

class DynamoServer {
 public:
  DynamoServer(ServerConfig config, GenerateHandler handler);
  ~DynamoServer();

  DynamoServer(const DynamoServer&) = delete;
  DynamoServer& operator=(const DynamoServer&) = delete;

  /// Start listening. Blocks the calling thread; intended to run in its own
  /// thread.
  void run();

  /// Stop the accept loop and close the listen socket. Safe to call from any
  /// thread.
  void shutdown();

  /// Bound port (valid after `run()` has bound, or 0 when not bound yet).
  uint16_t port() const { return actual_port_; }

  const ServerConfig& config() const { return config_; }

 private:
  ServerConfig config_;
  GenerateHandler handler_;
  int listen_fd_ = -1;
  uint16_t actual_port_ = 0;
  std::atomic<bool> running_{false};

  void handle_connection(int client_fd);
  bool read_exact(int fd, std::vector<uint8_t>& buf, size_t n);
  bool read_request(int fd, TcpRequestMessage& msg);
  void process_request(int fd, const TcpRequestMessage& msg);
  void stream_response(const TcpStreamConnectionInfo& conn_info,
                       const std::string& request_id,
                       const GenerateRequest& gen_req);
};

}  // namespace tt::dynamo
