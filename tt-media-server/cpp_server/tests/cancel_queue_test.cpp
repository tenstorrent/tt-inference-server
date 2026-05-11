// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "ipc/boost/cancel_queue.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace {

// Use PID-based unique names to avoid interference between parallel test runs.
std::string uniqueName(const std::string& base) {
  return base + "_" + std::to_string(getpid());
}

class CancelQueueTest : public ::testing::Test {
 protected:
  std::string queueName;

  void SetUp() override {
    queueName = uniqueName("test_cancel_q");
    tt::ipc::boost::CancelQueue::removeByName(queueName);
  }

  void TearDown() override {
    tt::ipc::boost::CancelQueue::removeByName(queueName);
  }
};

TEST_F(CancelQueueTest, PushAndPopRoundTrip) {
  tt::ipc::boost::CancelQueue queue(queueName, 16);

  uint32_t id1 = 123;
  uint32_t id2 = 456;
  queue.push(id1);
  queue.push(id2);

  std::vector<uint32_t> out;
  queue.tryPopAll(out);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], 123u);
  EXPECT_EQ(out[1], 456u);
}

TEST_F(CancelQueueTest, PopEmptyReturnsNothing) {
  tt::ipc::boost::CancelQueue queue(queueName, 16);

  std::vector<uint32_t> out;
  queue.tryPopAll(out);

  EXPECT_TRUE(out.empty());
}

TEST_F(CancelQueueTest, PushWhenFullDropsWithoutThrow) {
  tt::ipc::boost::CancelQueue queue(queueName, 2);

  queue.push(1);
  queue.push(2);
  // Queue is full — this should not throw, just log a warning and drop.
  EXPECT_NO_THROW(queue.push(3));

  std::vector<uint32_t> out;
  queue.tryPopAll(out);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], 1u);
  EXPECT_EQ(out[1], 2u);
}

TEST_F(CancelQueueTest, MultiplePushDrainInOrder) {
  tt::ipc::boost::CancelQueue queue(queueName, 64);

  for (int i = 0; i < 10; i++) {
    queue.push(static_cast<uint32_t>(i + 100));
  }

  std::vector<uint32_t> out;
  queue.tryPopAll(out);
  ASSERT_EQ(out.size(), 10u);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(out[i], static_cast<uint32_t>(i + 100));
  }
}

TEST_F(CancelQueueTest, PopAllThenPopAgainIsEmpty) {
  tt::ipc::boost::CancelQueue queue(queueName, 16);

  queue.push(999);

  std::vector<uint32_t> out;
  queue.tryPopAll(out);
  ASSERT_EQ(out.size(), 1u);

  out.clear();
  queue.tryPopAll(out);
  EXPECT_TRUE(out.empty());
}

TEST_F(CancelQueueTest, OpenExistingQueue) {
  // Create queue (simulates main process).
  tt::ipc::boost::CancelQueue creator(queueName, 16);
  creator.push(12345);

  // Open existing queue (simulates worker process).
  tt::ipc::boost::CancelQueue opener(queueName);

  std::vector<uint32_t> out;
  opener.tryPopAll(out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], 12345u);
}

TEST_F(CancelQueueTest, RemoveCleansUpResource) {
  {
    tt::ipc::boost::CancelQueue queue(queueName, 16);
    queue.remove();
  }
  // After removal, creating a new queue with the same name should succeed
  // (the old resource was cleaned up).
  EXPECT_NO_THROW(tt::ipc::boost::CancelQueue(queueName, 16));
}

}  // namespace
