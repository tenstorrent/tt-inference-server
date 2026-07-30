// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/layer_to_partition.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace tt::services {
namespace {

TEST(DeriveLayersPerPartition, CeilDividesAcrossWorkers) {
  EXPECT_EQ(deriveLayersPerPartition(/*numLayers=*/64, /*numPartitions=*/4),
            16u);
  EXPECT_EQ(deriveLayersPerPartition(/*numLayers=*/61, /*numPartitions=*/4),
            16u);
  EXPECT_EQ(deriveLayersPerPartition(/*numLayers=*/4, /*numPartitions=*/4), 1u);
}

TEST(DeriveLayersPerPartition, ZeroPartitionsReturnsZero) {
  EXPECT_EQ(deriveLayersPerPartition(/*numLayers=*/64, /*numPartitions=*/0),
            0u);
}

TEST(LayerToPartition, DisabledWhenLayersPerPartitionIsZero) {
  const LayerPartitionPolicy policy{.layersPerPartition = 0,
                                    .numPartitions = 4};
  EXPECT_FALSE(isLayerPartitionPolicyEnabled(policy));
  EXPECT_EQ(layerToPartition(/*layerId=*/0, policy), -1);
  EXPECT_EQ(layerToPartition(/*layerId=*/20, policy), -1);
}

TEST(LayerToPartition, DisabledWhenNumPartitionsIsZero) {
  const LayerPartitionPolicy policy{.layersPerPartition = 16,
                                    .numPartitions = 0};
  EXPECT_FALSE(isLayerPartitionPolicyEnabled(policy));
  EXPECT_EQ(layerToPartition(/*layerId=*/0, policy), -1);
}

TEST(LayerToPartition, MapsContiguousBlocksToOwners) {
  // Matches remote_kv_manager_e2e: 64 layers / 4 workers => 16 layers each.
  const LayerPartitionPolicy policy{.layersPerPartition = 16,
                                    .numPartitions = 4};
  EXPECT_TRUE(isLayerPartitionPolicyEnabled(policy));
  EXPECT_EQ(layerToPartition(/*layerId=*/0, policy), 0);
  EXPECT_EQ(layerToPartition(/*layerId=*/15, policy), 0);
  EXPECT_EQ(layerToPartition(/*layerId=*/16, policy), 1);
  EXPECT_EQ(layerToPartition(/*layerId=*/20, policy), 1);
  EXPECT_EQ(layerToPartition(/*layerId=*/31, policy), 1);
  EXPECT_EQ(layerToPartition(/*layerId=*/32, policy), 2);
  EXPECT_EQ(layerToPartition(/*layerId=*/48, policy), 3);
  EXPECT_EQ(layerToPartition(/*layerId=*/63, policy), 3);
}

TEST(LayerToPartition, OutOfRangeReturnsNegative) {
  const LayerPartitionPolicy policy{.layersPerPartition = 16,
                                    .numPartitions = 4};
  // layer 64 => partition 4, which is outside [0, 4).
  EXPECT_EQ(layerToPartition(/*layerId=*/64, policy), -1);
  EXPECT_EQ(layerToPartition(/*layerId=*/100, policy), -1);
}

TEST(LayerToPartition, SinglePartitionLegacySafe) {
  const LayerPartitionPolicy policy{.layersPerPartition = 64,
                                    .numPartitions = 1};
  EXPECT_EQ(layerToPartition(/*layerId=*/0, policy), 0);
  EXPECT_EQ(layerToPartition(/*layerId=*/63, policy), 0);
  EXPECT_EQ(layerToPartition(/*layerId=*/64, policy), -1);
}

TEST(MakeLayerToPartition, CallableMatchesDirectMapping) {
  const LayerPartitionPolicy policy{.layersPerPartition = 16,
                                    .numPartitions = 4};
  const auto mapFn = makeLayerToPartition(policy);
  for (uint32_t layerId = 0; layerId < 64; ++layerId) {
    EXPECT_EQ(mapFn(layerId), layerToPartition(layerId, policy)) << layerId;
  }
}

}  // namespace
}  // namespace tt::services
