// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "worker/worker_metrics_shm.hpp"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

namespace {

constexpr const char* TEST_SHM_NAME = "/tt_worker_metrics_test";
constexpr size_t NUM_WORKERS = 2;

bool shmExists(const std::string& name) {
  return access(("/dev/shm" + name).c_str(), F_OK) == 0;
}

TEST(WorkerMetricsShmTest, CreateOpenReadbackRoundtrip) {
  auto owner = tt::worker::WorkerMetricsShm::create(TEST_SHM_NAME, NUM_WORKERS);
  ASSERT_NE(owner, nullptr);
  ASSERT_EQ(owner->numWorkers(), NUM_WORKERS);

  auto reader = tt::worker::WorkerMetricsShm::open(TEST_SHM_NAME);
  ASSERT_NE(reader, nullptr);

  owner->setPid(0, 12345);
  owner->setLayout(0, tt::worker::MetricsLayout::SP_PIPELINE_RUNNER);
  owner->storeScratch(0, 0, 42);
  EXPECT_EQ(reader->loadScratch(0, 0), 42u);
  EXPECT_EQ(reader->layout(0), tt::worker::MetricsLayout::SP_PIPELINE_RUNNER);

  EXPECT_EQ(owner->fetchAddScratch(0, 1, 5), 0u);
  EXPECT_EQ(reader->loadScratch(0, 1), 5u);
  EXPECT_EQ(owner->fetchSubScratch(0, 1, 2), 5u);
  EXPECT_EQ(reader->loadScratch(0, 1), 3u);
}

TEST(WorkerMetricsShmTest, OwnerUnlinksOnDestruction) {
  {
    auto owner = tt::worker::WorkerMetricsShm::create(TEST_SHM_NAME, 1);
    ASSERT_NE(owner, nullptr);
    EXPECT_TRUE(shmExists(TEST_SHM_NAME));
  }
  EXPECT_FALSE(shmExists(TEST_SHM_NAME));
}

}  // namespace
