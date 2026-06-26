// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>
#include <string>

namespace tt::messaging {

/**
 * Polymorphic seam over KafkaConsumer so callers (e.g. RemoteKVManagerImpl)
 * can be unit-tested with an in-process fake instead of a real broker.
 *
 * Production code injects a tt::messaging::KafkaConsumer; tests inject a
 * fake that yields scripted ack messages.
 */
class IKafkaConsumer {
 public:
  virtual ~IKafkaConsumer() = default;

  /** See KafkaConsumer::receive. */
  virtual std::optional<std::string> receive(int timeoutMs) = 0;
};

}  // namespace tt::messaging
