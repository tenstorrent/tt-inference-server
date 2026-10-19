// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "config/settings.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/serialization.hpp"

namespace tt::ipc::embedding {

namespace ser = tt::ipc::serialization;

constexpr size_t EMBEDDING_MAX_TASK_MSG_BYTES = 64 * 1024;
constexpr size_t EMBEDDING_MAX_RESULT_MSG_BYTES = 20 * 1024;
constexpr size_t EMBEDDING_MAX_ERROR_BYTES = 512;

inline size_t maxTaskMessageBytes(size_t maxSeqLen) {
  (void)maxSeqLen;
  return EMBEDDING_MAX_TASK_MSG_BYTES;
}

inline size_t maxInputBytesForTaskMsg(size_t maxTaskMsgBytes) {
  return maxTaskMsgBytes > 512 ? maxTaskMsgBytes - 512 : 0;
}

struct EmbeddingIpcTask {
  uint32_t taskId = 0;
  uint32_t flags = 0;
  std::string model;
  std::string input;
  std::string user;

  static constexpr uint32_t FLAG_DONE = 1;
  static constexpr uint32_t FLAG_HAS_USER = 2;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }
  bool hasUser() const { return (flags & FLAG_HAS_USER) != 0; }

  static EmbeddingIpcTask done() {
    EmbeddingIpcTask task;
    task.flags = FLAG_DONE;
    return task;
  }

  static EmbeddingIpcTask fromDomain(const domain::EmbeddingRequest& req) {
    EmbeddingIpcTask task;
    task.taskId = req.task_id;
    task.model = req.model;
    task.input = req.input;
    if (req.user.has_value()) {
      task.flags |= FLAG_HAS_USER;
      task.user = *req.user;
    }
    return task;
  }

  domain::EmbeddingRequest toDomain() const {
    domain::EmbeddingRequest req(taskId);
    req.model = model;
    req.input = input;
    if (hasUser()) {
      req.user = user;
    }
    return req;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    ser::writeString(os, model);
    ser::writeString(os, input);
    ser::writeString(os, user);
  }

  static EmbeddingIpcTask deserialize(std::istream& is) {
    EmbeddingIpcTask task;
    is.read(reinterpret_cast<char*>(&task.taskId), sizeof(task.taskId));
    is.read(reinterpret_cast<char*>(&task.flags), sizeof(task.flags));
    task.model = ser::readString(is);
    task.input = ser::readString(is);
    task.user = ser::readString(is);
    return task;
  }
};

struct EmbeddingIpcResult {
  uint32_t taskId = 0;
  uint32_t flags = 0;
  int32_t totalTokens = 0;
  float batchCollectMs = 0.0f;
  std::string model;
  std::string error;
  std::vector<float> embedding;

  static constexpr uint32_t FLAG_DONE = 1;
  static constexpr uint32_t FLAG_CLAIMED = 2;
  static constexpr uint32_t FLAG_BATCH_END = 4;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }
  bool isClaimed() const { return (flags & FLAG_CLAIMED) != 0; }
  bool isBatchEnd() const { return (flags & FLAG_BATCH_END) != 0; }

  static EmbeddingIpcResult done() {
    EmbeddingIpcResult result;
    result.flags = FLAG_DONE;
    return result;
  }

  static EmbeddingIpcResult claimed(uint32_t taskIdVal, uint32_t batchSize = 1,
                                    float collectMs = 0.0f) {
    EmbeddingIpcResult result;
    result.taskId = taskIdVal;
    result.flags = FLAG_CLAIMED;
    result.totalTokens = static_cast<int32_t>(batchSize);
    result.batchCollectMs = collectMs;
    return result;
  }

  static EmbeddingIpcResult fromDomain(uint32_t taskIdVal,
                                       const domain::EmbeddingResponse& resp,
                                       bool batchEnd = false) {
    EmbeddingIpcResult out;
    out.taskId = taskIdVal;
    if (batchEnd) {
      out.flags |= FLAG_BATCH_END;
    }
    if (!resp.error.empty()) {
      out.error = resp.error.size() > EMBEDDING_MAX_ERROR_BYTES
                      ? resp.error.substr(0, EMBEDDING_MAX_ERROR_BYTES)
                      : resp.error;
    } else {
      out.totalTokens = resp.total_tokens;
      out.model = resp.model;
      out.embedding = resp.embedding;
    }
    return out;
  }

  domain::EmbeddingResponse toDomain() const {
    domain::EmbeddingResponse resp(taskId);
    resp.total_tokens = totalTokens;
    resp.model = model;
    resp.error = error;
    resp.embedding = embedding;
    return resp;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    os.write(reinterpret_cast<const char*>(&totalTokens), sizeof(totalTokens));
    os.write(reinterpret_cast<const char*>(&batchCollectMs),
             sizeof(batchCollectMs));
    ser::writeString(os, model);
    ser::writeString(os, error);
    ser::writeVector(os, embedding);
  }

  static EmbeddingIpcResult deserialize(std::istream& is) {
    EmbeddingIpcResult result;
    is.read(reinterpret_cast<char*>(&result.taskId), sizeof(result.taskId));
    is.read(reinterpret_cast<char*>(&result.flags), sizeof(result.flags));
    is.read(reinterpret_cast<char*>(&result.totalTokens),
            sizeof(result.totalTokens));
    is.read(reinterpret_cast<char*>(&result.batchCollectMs),
            sizeof(result.batchCollectMs));
    result.model = ser::readString(is);
    result.error = ser::readString(is);
    result.embedding = ser::readVector<float>(is);
    return result;
  }
};

class EmbeddingTaskQueue {
 public:
  using Queue =
      boost::MemoryQueue<EmbeddingIpcTask, EMBEDDING_MAX_TASK_MSG_BYTES>;

  EmbeddingTaskQueue(const std::string& name, int capacity,
                     size_t maxMsgSize = EMBEDDING_MAX_TASK_MSG_BYTES)
      : queue(std::make_unique<Queue>(name, capacity, maxMsgSize)) {}

  explicit EmbeddingTaskQueue(const std::string& name)
      : queue(Queue::openExisting(name)) {}

  void push(const EmbeddingIpcTask& task) { queue->push(task); }

  bool tryPop(EmbeddingIpcTask& out) { return queue->tryPop(out); }

  void receive(EmbeddingIpcTask& out) { queue->receive(out); }

  bool empty() const { return queue->empty(); }

  void remove() { queue->remove(); }

 private:
  std::unique_ptr<Queue> queue;
};

class EmbeddingResultQueue {
 public:
  using Queue =
      boost::MemoryQueue<EmbeddingIpcResult, EMBEDDING_MAX_RESULT_MSG_BYTES>;

  EmbeddingResultQueue(const std::string& name, int capacity)
      : queue(std::make_unique<Queue>(name, capacity)) {}

  explicit EmbeddingResultQueue(const std::string& name)
      : queue(Queue::openExisting(name)) {}

  void push(const EmbeddingIpcResult& result) { queue->push(result); }

  bool tryPop(EmbeddingIpcResult& out) { return queue->tryPop(out); }

  bool blockingPop(EmbeddingIpcResult& out) {
    queue->receive(out);
    return !out.isDone();
  }

  bool empty() const { return queue->empty(); }

  void shutdown() { queue->push(EmbeddingIpcResult::done()); }

  void remove() { queue->remove(); }

 private:
  std::unique_ptr<Queue> queue;
};

class EmbeddingQueueSet {
 public:
  std::shared_ptr<EmbeddingTaskQueue> taskQueue;
  std::vector<std::shared_ptr<EmbeddingResultQueue>> resultQueues;
  size_t maxTaskMsgBytes = EMBEDDING_MAX_TASK_MSG_BYTES;

  EmbeddingQueueSet(int numWorkers, size_t maxSeqLen, size_t maxBatchSize)
      : maxTaskMsgBytes(maxTaskMessageBytes(maxSeqLen)) {
    const int taskCap = static_cast<int>(tt::config::maxQueueSize());
    taskQueue = std::make_shared<EmbeddingTaskQueue>(
        tt::config::ttEmbeddingTaskQueueName(), taskCap, maxTaskMsgBytes);

    const int resultCap =
        static_cast<int>(std::max<size_t>(64, maxBatchSize * 4));
    resultQueues.reserve(static_cast<size_t>(numWorkers));
    for (int i = 0; i < numWorkers; ++i) {
      resultQueues.emplace_back(std::make_shared<EmbeddingResultQueue>(
          std::string(tt::config::ttEmbeddingResultQueueName()) +
              std::to_string(i),
          resultCap));
    }
  }

  ~EmbeddingQueueSet() { clear(); }

  void clear() {
    if (taskQueue) {
      taskQueue->remove();
      taskQueue.reset();
    }
    for (auto& queue : resultQueues) {
      queue->shutdown();
      queue->remove();
    }
    resultQueues.clear();
  }

  EmbeddingQueueSet(const EmbeddingQueueSet&) = delete;
  EmbeddingQueueSet& operator=(const EmbeddingQueueSet&) = delete;

  EmbeddingQueueSet(EmbeddingQueueSet&&) = default;
  EmbeddingQueueSet& operator=(EmbeddingQueueSet&&) = default;
};

}  // namespace tt::ipc::embedding
