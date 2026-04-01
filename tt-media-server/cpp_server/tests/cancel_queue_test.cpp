// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "domain/task_id.hpp"
#include "ipc/boost_ipc_cancel_queue.hpp"

namespace {

// Use PID-based unique names to avoid interference between parallel test runs.
std::string uniqueName(const std::string& base) {
  return base + "_" + std::to_string(getpid());
}

class BoostIpcCancelQueueTest : public ::testing::Test {
 protected:
  std::string queueName;

  void SetUp() override {
    queueName = uniqueName("test_cancel_q");
    tt::ipc::BoostIpcCancelQueue::removeByName(queueName);
  }

  void TearDown() override {
    tt::ipc::BoostIpcCancelQueue::removeByName(queueName);
  }
};

TEST_F(BoostIpcCancelQueueTest, PushAndPopRoundTrip) {
  tt::ipc::BoostIpcCancelQueue queue(queueName, 16);

  tt::domain::TaskID id1("task-abc-123");
  tt::domain::TaskID id2("task-def-456");
  queue.push(id1);
  queue.push(id2);

  std::vector<tt::domain::TaskID> out;
  queue.tryPopAll(out);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].id, "task-abc-123");
  EXPECT_EQ(out[1].id, "task-def-456");
}

TEST_F(BoostIpcCancelQueueTest, PopEmptyReturnsNothing) {
  tt::ipc::BoostIpcCancelQueue queue(queueName, 16);

  std::vector<tt::domain::TaskID> out;
  queue.tryPopAll(out);

  EXPECT_TRUE(out.empty());
}

TEST_F(BoostIpcCancelQueueTest, PushWhenFullDropsWithoutThrow) {
  tt::ipc::BoostIpcCancelQueue queue(queueName, 2);

  queue.push(tt::domain::TaskID("t1"));
  queue.push(tt::domain::TaskID("t2"));
  // Queue is full — this should not throw, just log a warning and drop.
  EXPECT_NO_THROW(queue.push(tt::domain::TaskID("t3")));

  std::vector<tt::domain::TaskID> out;
  queue.tryPopAll(out);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].id, "t1");
  EXPECT_EQ(out[1].id, "t2");
}

TEST_F(BoostIpcCancelQueueTest, MultiplePushDrainInOrder) {
  tt::ipc::BoostIpcCancelQueue queue(queueName, 64);

  for (int i = 0; i < 10; i++) {
    queue.push(tt::domain::TaskID("id-" + std::to_string(i)));
  }

  std::vector<tt::domain::TaskID> out;
  queue.tryPopAll(out);
  ASSERT_EQ(out.size(), 10u);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(out[i].id, "id-" + std::to_string(i));
  }
}

TEST_F(BoostIpcCancelQueueTest, PopAllThenPopAgainIsEmpty) {
  tt::ipc::BoostIpcCancelQueue queue(queueName, 16);

  queue.push(tt::domain::TaskID("x"));

  std::vector<tt::domain::TaskID> out;
  queue.tryPopAll(out);
  ASSERT_EQ(out.size(), 1u);

  out.clear();
  queue.tryPopAll(out);
  EXPECT_TRUE(out.empty());
}

TEST_F(BoostIpcCancelQueueTest, OpenExistingQueue) {
  // Create queue (simulates main process).
  tt::ipc::BoostIpcCancelQueue creator(queueName, 16);
  creator.push(tt::domain::TaskID("from-main"));

  // Open existing queue (simulates worker process).
  tt::ipc::BoostIpcCancelQueue opener(queueName);

  std::vector<tt::domain::TaskID> out;
  opener.tryPopAll(out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, "from-main");
}

TEST_F(BoostIpcCancelQueueTest, RemoveCleansUpResource) {
  {
    tt::ipc::BoostIpcCancelQueue queue(queueName, 16);
    queue.remove();
  }
  // After removal, creating a new queue with the same name should succeed
  // (the old resource was cleaned up).
  EXPECT_NO_THROW(tt::ipc::BoostIpcCancelQueue(queueName, 16));
}

}  // namespace
