// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "ipc/media_payload_ipc.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(MediaPayloadIpcTest, TaskQueueRoundTripsPayloadPaths) {
  const std::string queueName = "test_media_payload_tasks";
  tt::ipc::media_payload::MediaPayloadTaskQueue::Queue::remove(queueName);

  tt::ipc::media_payload::MediaPayloadTaskQueue owner(queueName, 4);
  tt::ipc::media_payload::MediaPayloadTaskQueue peer(queueName);

  tt::ipc::media_payload::MediaPayloadTask task;
  task.task_id = 42;
  task.request_path = "/tmp/request-42.json";
  task.response_path = "/tmp/response-42.json";
  owner.push(task);

  tt::ipc::media_payload::MediaPayloadTask received;
  peer.receive(received);

  EXPECT_EQ(received.task_id, task.task_id);
  EXPECT_EQ(received.request_path, task.request_path);
  EXPECT_EQ(received.response_path, task.response_path);

  owner.remove();
}

TEST(MediaPayloadIpcTest, ResultQueueUsesDonePillForShutdown) {
  const std::string queueName = "test_media_payload_results";
  tt::ipc::media_payload::MediaPayloadResultQueue::Queue::remove(queueName);

  tt::ipc::media_payload::MediaPayloadResultQueue owner(queueName, 4);
  tt::ipc::media_payload::MediaPayloadResultQueue peer(queueName);

  tt::ipc::media_payload::MediaPayloadResult result;
  result.task_id = 7;
  result.response_path = "/tmp/response-7.json";
  result.generation_time_seconds = 1.25;
  ASSERT_TRUE(owner.push(result));

  tt::ipc::media_payload::MediaPayloadResult received;
  EXPECT_TRUE(peer.blockingPop(received));
  EXPECT_EQ(received.task_id, result.task_id);
  EXPECT_EQ(received.response_path, result.response_path);
  EXPECT_DOUBLE_EQ(received.generation_time_seconds,
                   result.generation_time_seconds);

  owner.shutdown();
  EXPECT_FALSE(peer.blockingPop(received));

  owner.remove();
}
