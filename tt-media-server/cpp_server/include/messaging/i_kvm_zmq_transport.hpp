// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace tt::messaging {

/**
 * Polymorphic interface over the ZMQ command channel to kv_manager so
 * `RemoteKVManagerZmqImpl` can be unit-tested with an in-process fake
 * instead of standing up real ZMQ sockets. Combines the send side (analogue
 * of `IKafkaProducer`) and the receive side (analogue of `IKafkaConsumer`)
 * because commands and acknowledgements share one DEALER connection.
 */
class IKvmZmqTransport {
 public:
  virtual ~IKvmZmqTransport() = default;

  /**
   * Send a command payload to kv_manager. Returns true on success. On
   * failure, populates `errorMessage` (if non-null) with a human-readable
   * reason. Implementations may queue the payload for their receive/pump
   * thread. Analogous to `IKafkaProducer::send`.
   */
  virtual bool send(std::string_view payload,
                    std::string* errorMessage = nullptr) = 0;

  /**
   * Receive one ack payload from kv_manager, blocking up to `timeoutMs`.
   * Returns std::nullopt if the timeout fires with nothing in the queue.
   * Analogous to `IKafkaConsumer::receive`.
   */
  virtual std::optional<std::string> receive(int timeoutMs) = 0;
};

}  // namespace tt::messaging
