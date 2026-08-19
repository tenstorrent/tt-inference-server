// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/voice_sample_cache.hpp"

#include <gtest/gtest.h>

namespace tt::utils {
namespace {

TEST(VoiceSampleCacheTest, HitRefreshesAccessOrder) {
  VoiceSampleCache cache(2);
  const VoiceSampleCache::Samples first = {1, 2};
  const VoiceSampleCache::Samples second = {3, 4};
  const VoiceSampleCache::Samples third = {5, 6};
  cache.add(first, {10});
  cache.add(second, {20});
  EXPECT_EQ(cache.get(first), (VoiceSampleCache::SpeechIds{10}));
  cache.add(third, {30});
  EXPECT_TRUE(cache.exists(first));
  EXPECT_FALSE(cache.exists(second));
}

TEST(VoiceSampleCacheTest, ReplacesExistingEntryWithoutEviction) {
  VoiceSampleCache cache(2);
  const VoiceSampleCache::Samples first = {1};
  const VoiceSampleCache::Samples second = {2};
  cache.add(first, {10});
  cache.add(second, {20});
  cache.add(first, {11, 12});
  EXPECT_EQ(cache.get(first), (VoiceSampleCache::SpeechIds{11, 12}));
  EXPECT_EQ(cache.get(second), (VoiceSampleCache::SpeechIds{20}));
}

TEST(VoiceSampleCacheTest, EvictsLeastRecentlyUsedEntry) {
  VoiceSampleCache cache(2);
  const VoiceSampleCache::Samples first = {1};
  const VoiceSampleCache::Samples second = {2};
  const VoiceSampleCache::Samples third = {3};
  cache.add(first, {10});
  cache.add(second, {20});
  cache.add(third, {30});
  EXPECT_FALSE(cache.exists(first));
  EXPECT_TRUE(cache.exists(second));
  EXPECT_TRUE(cache.exists(third));
}

TEST(VoiceSampleCacheTest, ZeroCapacityDoesNotStoreEntries) {
  VoiceSampleCache cache(0);
  const VoiceSampleCache::Samples samples = {1};
  cache.add(samples, {10});
  EXPECT_FALSE(cache.exists(samples));
}

}  // namespace
}  // namespace tt::utils
