// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace tt::services {

/**
 * Exclusive Kafka ownership pin for N-prefill migration workers (#4795).
 *
 * Modes:
 *   - pin set  => consumer uses rd_kafka_assign(partition); no rebalance.
 *                 Prefill workers share one group id (group is required by
 *                 librdkafka but does not own partition assignment).
 *   - pin unset => consumer uses subscribe/rebalance (legacy broadcast when
 *                 each prefill has its own group id).
 *
 * Inference publishes to f(layer); the worker pinned to that partition is the
 * sole owner of the request. Decode peer lists (WORKER_PEERS) remain the
 * data-plane adjacency list and must stay consistent with this pin map.
 *
 * Ack affinity uses the same partition index on the ack topic, so both request
 * and ack topics must be created with partitions >= N (Phase 3 / migration_cli).
 */
inline std::optional<int32_t> parseKafkaPartitionPin(std::string_view raw) {
  if (raw.empty()) {
    return std::nullopt;
  }
  try {
    std::size_t consumed = 0;
    const long value = std::stol(std::string(raw), &consumed, 10);
    if (consumed != raw.size() || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int32_t>::max())) {
      return std::nullopt;
    }
    return static_cast<int32_t>(value);
  } catch (...) {
    return std::nullopt;
  }
}

/** True when raw looks set but parseKafkaPartitionPin rejected it. */
inline bool isInvalidKafkaPartitionPin(std::string_view raw) {
  return !raw.empty() && !parseKafkaPartitionPin(raw).has_value();
}

/**
 * Default request-consumer group id for a prefill worker.
 * Pinned ownership shares one group; broadcast mode uses a per-worker group so
 * every prefill receives every request (legacy N=1 workaround).
 */
inline std::string defaultPrefillKafkaGroupId(std::string_view workerTag,
                                              bool hasPartitionPin,
                                              std::string_view sharedGroupId) {
  if (hasPartitionPin) {
    return std::string(sharedGroupId);
  }
  return std::string(sharedGroupId) + "-prefill-" + std::string(workerTag);
}

}  // namespace tt::services
