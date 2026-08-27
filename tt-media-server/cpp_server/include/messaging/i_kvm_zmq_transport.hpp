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
 * because ZMQ is inherently paired: our PUB and SUB share one context and
 * the same lifetime.
 *
 * Wire framing on the ZMQ path: every message is a 2-frame multipart:
 *   [ topic (byte prefix) ][ JSON payload ]
 * kv_manager's SUB filters on the topic prefix and echoes it back on the
 * ack. The topic string is chosen by the transport implementation (see
 * `KvmZmqTransport`) so callers work in terms of the payload only.
 */
class IKvmZmqTransport {
 public:
  virtual ~IKvmZmqTransport() = default;

  /**
   * Publish a command payload to kv_manager. Returns true on success. On
   * failure, populates `errorMessage` (if non-null) with a human-readable
   * reason. Analogous to `IKafkaProducer::send`.
   *
   * Non-blocking: the underlying PUB socket queues the message. There is no
   * end-to-end delivery guarantee — a subscriber that is not yet connected
   * will silently drop messages (the classic PUB/SUB slow-joiner problem);
   * we mitigate that by sizing the connection wait in `KvmZmqTransport::open`.
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
