// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>

namespace tt::services {

/**
 * Shared layer -> Kafka partition ownership policy for N-prefill KV migration
 * (#4795). Inference publishes to partition f(layer); each prefill worker
 * assigns exactly one partition. One source of truth for both sides.
 *
 * Mapping: partition = layerId / layersPerPartition, clamped to
 * [0, numPartitions). A disabled or out-of-range result is -1, matching
 * RemoteKVManagerImpl (negative => do not force a partition).
 */
struct LayerPartitionPolicy {
  // Contiguous layer block owned by one Kafka partition. 0 disables routing.
  uint32_t layersPerPartition = 0;
  // Request-topic partition count. Must be > 0 when the policy is enabled.
  uint32_t numPartitions = 1;
};

inline bool isLayerPartitionPolicyEnabled(
    const LayerPartitionPolicy& policy) {
  return policy.layersPerPartition > 0 && policy.numPartitions > 0;
}

/** Ceil(numLayers / numPartitions). Returns 0 when numPartitions == 0. */
inline uint32_t deriveLayersPerPartition(uint32_t numLayers,
                                         uint32_t numPartitions) {
  if (numPartitions == 0) {
    return 0;
  }
  return (numLayers + numPartitions - 1) / numPartitions;
}

/**
 * Maps layerId to its owning Kafka partition.
 * @return partition in [0, numPartitions), or -1 if disabled / out of range.
 */
inline int32_t layerToPartition(uint32_t layerId,
                                const LayerPartitionPolicy& policy) {
  if (!isLayerPartitionPolicyEnabled(policy)) {
    return -1;
  }
  const uint32_t partition = layerId / policy.layersPerPartition;
  if (partition >= policy.numPartitions) {
    return -1;
  }
  return static_cast<int32_t>(partition);
}

/** Callable form for RemoteKVManagerImpl::LayerToPartition. */
inline std::function<int32_t(uint32_t)> makeLayerToPartition(
    LayerPartitionPolicy policy) {
  return [policy](uint32_t layerId) {
    return layerToPartition(layerId, policy);
  };
}

}  // namespace tt::services
