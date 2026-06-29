// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <string_view>

namespace tt::messaging {

/**
 * Polymorphic seam over KafkaProducer so callers (e.g. RemoteKVManagerImpl)
 * can be unit-tested with an in-process fake instead of a real broker.
 *
 * Production code injects a tt::messaging::KafkaProducer; tests inject a
 * fake that records sent payloads.
 */
class IKafkaProducer {
 public:
  virtual ~IKafkaProducer() = default;

  /** See KafkaProducer::send. */
  virtual bool send(std::string_view payload, std::string* errorMessage) = 0;

  /** See KafkaProducer::flush. */
  virtual bool flush(int timeoutMs, std::string* errorMessage) = 0;
};

}  // namespace tt::messaging
