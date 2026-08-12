// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/kafka_partition_pin.hpp"

#include <gtest/gtest.h>

namespace tt::services {
namespace {

TEST(ParseKafkaPartitionPin, EmptyMeansUnset) {
  EXPECT_FALSE(parseKafkaPartitionPin("").has_value());
  EXPECT_FALSE(isInvalidKafkaPartitionPin(""));
}

TEST(ParseKafkaPartitionPin, AcceptsNonNegativeIntegers) {
  EXPECT_EQ(parseKafkaPartitionPin("0"), 0);
  EXPECT_EQ(parseKafkaPartitionPin("3"), 3);
  EXPECT_EQ(parseKafkaPartitionPin("42"), 42);
}

TEST(ParseKafkaPartitionPin, RejectsInvalidValues) {
  EXPECT_FALSE(parseKafkaPartitionPin("-1").has_value());
  EXPECT_TRUE(isInvalidKafkaPartitionPin("-1"));
  EXPECT_FALSE(parseKafkaPartitionPin("1.5").has_value());
  EXPECT_TRUE(isInvalidKafkaPartitionPin("1.5"));
  EXPECT_FALSE(parseKafkaPartitionPin("abc").has_value());
  EXPECT_TRUE(isInvalidKafkaPartitionPin("abc"));
  EXPECT_FALSE(parseKafkaPartitionPin("1x").has_value());
  EXPECT_TRUE(isInvalidKafkaPartitionPin("1x"));
}

TEST(DefaultPrefillKafkaGroupId, SharedWhenPinned) {
  EXPECT_EQ(defaultPrefillKafkaGroupId("prefill-0", /*hasPartitionPin=*/true,
                                       "migration-workers"),
            "migration-workers");
}

TEST(DefaultPrefillKafkaGroupId, PerWorkerWhenBroadcast) {
  EXPECT_EQ(defaultPrefillKafkaGroupId("prefill-0", /*hasPartitionPin=*/false,
                                       "migration-workers"),
            "migration-workers-prefill-prefill-0");
}

}  // namespace
}  // namespace tt::services
