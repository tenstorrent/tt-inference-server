// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/affinity_cache.hpp"

#include <gtest/gtest.h>

namespace tt::gateway {
namespace {

TEST(AffinityCacheTest, LookupOnEmptyCacheReturnsNullopt) {
  AffinityCache cache;
  EXPECT_FALSE(cache.lookup(42).has_value());
}

TEST(AffinityCacheTest, RecordThenLookup) {
  AffinityCache cache;
  cache.record(/*hash=*/42, "prefill-A");

  auto hit = cache.lookup(42);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(*hit, "prefill-A");
}

TEST(AffinityCacheTest, RecordIgnoresZeroHash) {
  AffinityCache cache;
  cache.record(/*hash=*/0, "prefill-A");
  EXPECT_FALSE(cache.lookup(0).has_value());
}

TEST(AffinityCacheTest, RecordOverwritesExistingEntry) {
  AffinityCache cache;
  cache.record(42, "prefill-A");
  cache.record(42, "prefill-B");

  auto hit = cache.lookup(42);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(*hit, "prefill-B");
}

TEST(AffinityCacheTest, EvictPrefillRemovesAllEntriesForThatPrefill) {
  AffinityCache cache;
  cache.record(1, "prefill-A");
  cache.record(2, "prefill-B");
  cache.record(3, "prefill-A");
  cache.record(4, "prefill-C");

  cache.evictPrefill("prefill-A");

  EXPECT_FALSE(cache.lookup(1).has_value());
  EXPECT_FALSE(cache.lookup(3).has_value());
  EXPECT_TRUE(cache.lookup(2).has_value());
  EXPECT_TRUE(cache.lookup(4).has_value());
}

TEST(AffinityCacheTest, EvictPrefillIsNoopForUnknownId) {
  AffinityCache cache;
  cache.record(1, "prefill-A");
  cache.evictPrefill("prefill-DOES-NOT-EXIST");

  auto hit = cache.lookup(1);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(*hit, "prefill-A");
}

}  // namespace
}  // namespace tt::gateway
