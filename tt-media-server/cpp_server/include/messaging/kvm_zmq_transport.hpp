// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "messaging/i_kvm_zmq_transport.hpp"

namespace tt::messaging {

/**
 * Configuration for the ZMQ transport to kv_manager. Endpoints are ZMQ URIs
 * (e.g. "tcp://0.0.0.0:5555"). Because kv_manager's `ZmqCommandTransport`
 * always `zmq_connect`s on both its SUB and PUB sockets, this side must
 * `zmq_bind`. The two endpoints intentionally use different ports so PUB
 * and SUB flows do not conflict.
 *
 * `topic` is the byte prefix used to filter on kv_manager's SUB
 * (`ZMQ_SUBSCRIBE`) and is echoed back on the ack. Match kv_manager's
 * `KV_MANAGER_ZMQ_TOPICS` (defaults to "L1").
 */
struct KvmZmqTransportConfig {
  std::string cmdEndpoint;    // We bind PUB here; kv_manager connects SUB.
  std::string replyEndpoint;  // We bind SUB here; kv_manager connects PUB.
  std::string topic;          // Byte prefix (default: "L1").
};

/**
 * cppzmq-backed implementation of `IKvmZmqTransport`. Owns a single ZMQ
 * context, a bound PUB socket for commands, and a bound SUB socket for
 * acks. Both sockets are set `ZMQ_LINGER=0` so the destructor never blocks
 * shutdown on pending unacknowledged messages.
 *
 * Endpoint model reminder (see the plan doc):
 *   tt-inference-server (us):  PUB bind cmd  |  SUB bind reply
 *   kv_manager (them):         SUB connect   |  PUB connect
 * This lets us start first and stay up regardless of kv_manager restarts.
 */
class KvmZmqTransport : public IKvmZmqTransport {
 public:
  explicit KvmZmqTransport(KvmZmqTransportConfig config);
  ~KvmZmqTransport() override;

  KvmZmqTransport(const KvmZmqTransport&) = delete;
  KvmZmqTransport& operator=(const KvmZmqTransport&) = delete;

  bool send(std::string_view payload,
            std::string* errorMessage = nullptr) override;

  std::optional<std::string> receive(int timeoutMs) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace tt::messaging
