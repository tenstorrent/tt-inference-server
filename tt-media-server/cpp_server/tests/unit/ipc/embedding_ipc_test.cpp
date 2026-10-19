// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "ipc/embedding_ipc.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

namespace tt::ipc::embedding {
namespace {

TEST(EmbeddingIpcTest, TaskSerializationRoundTripsDomainRequest) {
  domain::EmbeddingRequest req(101);
  req.model = "BAAI/bge-large-en-v1.5";
  req.input = "hello embedding world";
  req.user = "alice";

  auto task = EmbeddingIpcTask::fromDomain(req);
  EXPECT_FALSE(task.isDone());
  EXPECT_TRUE(task.hasUser());

  std::stringstream ss;
  task.serialize(ss);
  auto decoded = EmbeddingIpcTask::deserialize(ss);

  auto domainOut = decoded.toDomain();
  EXPECT_EQ(domainOut.task_id, 101u);
  EXPECT_EQ(domainOut.model, "BAAI/bge-large-en-v1.5");
  EXPECT_EQ(domainOut.input, "hello embedding world");
  ASSERT_TRUE(domainOut.user.has_value());
  EXPECT_EQ(*domainOut.user, "alice");
}

TEST(EmbeddingIpcTest, TaskSerializationRoundTripsOptionalUserUnset) {
  domain::EmbeddingRequest req(42);
  req.model = "BAAI/bge-m3";
  req.input = "no user";

  auto task = EmbeddingIpcTask::fromDomain(req);
  EXPECT_FALSE(task.hasUser());

  std::stringstream ss;
  task.serialize(ss);
  auto decoded = EmbeddingIpcTask::deserialize(ss);

  auto domainOut = decoded.toDomain();
  EXPECT_EQ(domainOut.task_id, 42u);
  EXPECT_FALSE(domainOut.user.has_value());
}

TEST(EmbeddingIpcTest, ResultSerializationRoundTripsVectorAndError) {
  domain::EmbeddingResponse okResp(7);
  okResp.model = "BAAI/bge-large-en-v1.5";
  okResp.total_tokens = 12;
  okResp.embedding = {0.25f, -0.5f, 0.75f, 1.0f};

  auto ipcOk = EmbeddingIpcResult::fromDomain(7, okResp);
  std::stringstream ssOk;
  ipcOk.serialize(ssOk);
  auto decodedOk = EmbeddingIpcResult::deserialize(ssOk).toDomain();
  EXPECT_EQ(decodedOk.task_id, 7u);
  EXPECT_EQ(decodedOk.model, "BAAI/bge-large-en-v1.5");
  EXPECT_EQ(decodedOk.total_tokens, 12);
  EXPECT_EQ(decodedOk.embedding, okResp.embedding);
  EXPECT_TRUE(decodedOk.error.empty());

  domain::EmbeddingResponse errResp(8);
  errResp.error = "model mismatch";
  auto ipcErr = EmbeddingIpcResult::fromDomain(8, errResp);
  std::stringstream ssErr;
  ipcErr.serialize(ssErr);
  auto decodedErr = EmbeddingIpcResult::deserialize(ssErr).toDomain();
  EXPECT_EQ(decodedErr.task_id, 8u);
  EXPECT_EQ(decodedErr.error, "model mismatch");
  EXPECT_TRUE(decodedErr.embedding.empty());
}

TEST(EmbeddingIpcTest, TaskAndResultQueuesRoundTripAndClaimSignal) {
  const std::string taskQueueName = "test_embedding_task_queue";
  const std::string resultQueueName = "test_embedding_result_queue";
  EmbeddingTaskQueue::Queue::remove(taskQueueName);
  EmbeddingResultQueue::Queue::remove(resultQueueName);

  EmbeddingTaskQueue taskOwner(taskQueueName, 8, maxTaskMessageBytes(512));
  EmbeddingTaskQueue taskPeer(taskQueueName);

  domain::EmbeddingRequest req(55);
  req.model = "BAAI/bge-large-en-v1.5";
  req.input = "queued text";
  taskOwner.push(EmbeddingIpcTask::fromDomain(req));

  EmbeddingIpcTask receivedTask;
  EXPECT_TRUE(taskPeer.tryPop(receivedTask));
  EXPECT_EQ(receivedTask.taskId, 55u);
  EXPECT_EQ(receivedTask.input, "queued text");

  EmbeddingResultQueue resultOwner(resultQueueName, 8);
  EmbeddingResultQueue resultPeer(resultQueueName);

  resultPeer.push(EmbeddingIpcResult::claimed(55, 4));
  EmbeddingIpcResult receivedResult;
  ASSERT_TRUE(resultOwner.blockingPop(receivedResult));
  EXPECT_TRUE(receivedResult.isClaimed());
  EXPECT_EQ(receivedResult.taskId, 55u);
  EXPECT_EQ(receivedResult.totalTokens, 4);

  domain::EmbeddingResponse resp(55);
  resp.model = req.model;
  resp.total_tokens = 3;
  resp.embedding = {1.0f, 0.0f};
  resultPeer.push(EmbeddingIpcResult::fromDomain(55, resp));

  ASSERT_TRUE(resultOwner.blockingPop(receivedResult));
  EXPECT_FALSE(receivedResult.isClaimed());
  EXPECT_EQ(receivedResult.taskId, 55u);
  EXPECT_EQ(receivedResult.embedding.size(), 2u);

  resultOwner.shutdown();
  EXPECT_FALSE(resultOwner.blockingPop(receivedResult));

  taskOwner.remove();
  resultOwner.remove();
}

}  // namespace
}  // namespace tt::ipc::embedding
