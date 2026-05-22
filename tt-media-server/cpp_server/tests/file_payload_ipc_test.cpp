// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "ipc/file_payload_ipc.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(FilePayloadIpcTest, TaskQueueRoundTripsPayloadPaths) {
  const std::string queueName = "test_file_payload_tasks";
  tt::ipc::file_payload::FilePayloadTaskQueue::Queue::remove(queueName);

  tt::ipc::file_payload::FilePayloadTaskQueue owner(queueName, 4);
  tt::ipc::file_payload::FilePayloadTaskQueue peer(queueName);

  tt::ipc::file_payload::FilePayloadTask task;
  task.task_id = 42;
  task.request_path = "/tmp/request-42.json";
  task.response_path = "/tmp/response-42.json";
  owner.push(task);

  tt::ipc::file_payload::FilePayloadTask received;
  peer.receive(received);

  EXPECT_EQ(received.task_id, task.task_id);
  EXPECT_EQ(received.request_path, task.request_path);
  EXPECT_EQ(received.response_path, task.response_path);

  owner.remove();
}

TEST(FilePayloadIpcTest, ResultQueueUsesDonePillForShutdown) {
  const std::string queueName = "test_file_payload_results";
  tt::ipc::file_payload::FilePayloadResultQueue::Queue::remove(queueName);

  tt::ipc::file_payload::FilePayloadResultQueue owner(queueName, 4);
  tt::ipc::file_payload::FilePayloadResultQueue peer(queueName);

  tt::ipc::file_payload::FilePayloadResult result;
  result.task_id = 7;
  result.response_path = "/tmp/response-7.json";
  result.generation_time_seconds = 1.25;
  ASSERT_TRUE(owner.push(result));

  tt::ipc::file_payload::FilePayloadResult received;
  EXPECT_TRUE(peer.blockingPop(received));
  EXPECT_EQ(received.task_id, result.task_id);
  EXPECT_EQ(received.response_path, result.response_path);
  EXPECT_DOUBLE_EQ(received.generation_time_seconds,
                   result.generation_time_seconds);

  owner.shutdown();
  EXPECT_FALSE(peer.blockingPop(received));

  owner.remove();
}
