// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "messaging/i_kvm_zmq_transport.hpp"

namespace tt::messaging {

/** Configuration for the direct command channel to kv_manager's ROUTER. */
struct KvmZmqTransportConfig {
  std::string endpoint;
};

/**
 * cppzmq-backed implementation of `IKvmZmqTransport`. A DEALER connects to
 * kv_manager's ROUTER and exchanges single-frame JSON commands and acks.
 * send() queues commands so the receive thread exclusively owns the socket.
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
