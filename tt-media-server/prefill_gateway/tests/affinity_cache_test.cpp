// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/affinity_cache.hpp"

#include <gtest/gtest.h>

namespace tt::gateway {
namespace {

TEST(AffinityCacheTest, EmptyOnStartup) {
  AffinityCache cache;
  EXPECT_EQ(cache.size(), 0u);
  EXPECT_FALSE(cache.lookup(42).has_value());
}

TEST(AffinityCacheTest, RecordThenLookup) {
  AffinityCache cache;
  cache.record(/*hash=*/42, "prefill-A");

  auto hit = cache.lookup(42);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(*hit, "prefill-A");
  EXPECT_EQ(cache.size(), 1u);
}

TEST(AffinityCacheTest, RecordIgnoresZeroHash) {
  AffinityCache cache;
  cache.record(/*hash=*/0, "prefill-A");

  EXPECT_FALSE(cache.lookup(0).has_value());
  EXPECT_EQ(cache.size(), 0u);
}

TEST(AffinityCacheTest, RecordOverwritesExistingEntry) {
  AffinityCache cache;
  cache.record(42, "prefill-A");
  cache.record(42, "prefill-B");

  auto hit = cache.lookup(42);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(*hit, "prefill-B");
  EXPECT_EQ(cache.size(), 1u);
}

TEST(AffinityCacheTest, EvictPrefillRemovesAllEntriesForThatPrefill) {
  AffinityCache cache;
  cache.record(1, "prefill-A");
  cache.record(2, "prefill-B");
  cache.record(3, "prefill-A");
  cache.record(4, "prefill-C");
  EXPECT_EQ(cache.size(), 4u);

  cache.evictPrefill("prefill-A");

  EXPECT_EQ(cache.size(), 2u);
  EXPECT_FALSE(cache.lookup(1).has_value());
  EXPECT_FALSE(cache.lookup(3).has_value());
  EXPECT_TRUE(cache.lookup(2).has_value());
  EXPECT_TRUE(cache.lookup(4).has_value());
}

TEST(AffinityCacheTest, EvictPrefillIsNoopForUnknownId) {
  AffinityCache cache;
  cache.record(1, "prefill-A");
  cache.evictPrefill("prefill-DOES-NOT-EXIST");

  EXPECT_EQ(cache.size(), 1u);
  EXPECT_TRUE(cache.lookup(1).has_value());
}

TEST(AffinityCacheTest, EvictHashDropsSingleEntry) {
  AffinityCache cache;
  cache.record(1, "prefill-A");
  cache.record(2, "prefill-A");

  cache.evictHash(1);

  EXPECT_FALSE(cache.lookup(1).has_value());
  EXPECT_TRUE(cache.lookup(2).has_value());
  EXPECT_EQ(cache.size(), 1u);
}

TEST(AffinityCacheTest, EvictHashIsNoopForUnknownHash) {
  AffinityCache cache;
  cache.record(1, "prefill-A");
  cache.evictHash(999);

  EXPECT_EQ(cache.size(), 1u);
}

}  // namespace
}  // namespace tt::gateway
