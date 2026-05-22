// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "config/settings.hpp"
#include "ipc/boost/boost_memory_queue.hpp"

namespace tt::ipc::file_payload {

namespace detail {

inline void writeString(std::ostream& os, const std::string& value) {
  const uint32_t size = static_cast<uint32_t>(value.size());
  os.write(reinterpret_cast<const char*>(&size), sizeof(size));
  os.write(value.data(), static_cast<std::streamsize>(value.size()));
}

inline std::string readString(std::istream& is) {
  uint32_t size = 0;
  is.read(reinterpret_cast<char*>(&size), sizeof(size));
  std::string value(size, '\0');
  if (size > 0) {
    is.read(value.data(), static_cast<std::streamsize>(size));
  }
  return value;
}

}  // namespace detail

struct FilePayloadTask {
  uint32_t task_id = 0;
  uint32_t flags = 0;
  std::string request_path;
  std::string response_path;

  static constexpr uint32_t FLAG_DONE = 1;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }

  static FilePayloadTask done() {
    FilePayloadTask task;
    task.flags = FLAG_DONE;
    return task;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&task_id), sizeof(task_id));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    detail::writeString(os, request_path);
    detail::writeString(os, response_path);
  }

  static FilePayloadTask deserialize(std::istream& is) {
    FilePayloadTask task;
    is.read(reinterpret_cast<char*>(&task.task_id), sizeof(task.task_id));
    is.read(reinterpret_cast<char*>(&task.flags), sizeof(task.flags));
    task.request_path = detail::readString(is);
    task.response_path = detail::readString(is);
    return task;
  }
};

struct FilePayloadResult {
  uint32_t task_id = 0;
  uint32_t flags = 0;
  double generation_time_seconds = 0.0;
  std::string response_path;
  std::string error;

  static constexpr uint32_t FLAG_DONE = 1;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }

  static FilePayloadResult done() {
    FilePayloadResult result;
    result.flags = FLAG_DONE;
    return result;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&task_id), sizeof(task_id));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    os.write(reinterpret_cast<const char*>(&generation_time_seconds),
             sizeof(generation_time_seconds));
    detail::writeString(os, response_path);
    detail::writeString(os, error);
  }

  static FilePayloadResult deserialize(std::istream& is) {
    FilePayloadResult result;
    is.read(reinterpret_cast<char*>(&result.task_id), sizeof(result.task_id));
    is.read(reinterpret_cast<char*>(&result.flags), sizeof(result.flags));
    is.read(reinterpret_cast<char*>(&result.generation_time_seconds),
            sizeof(result.generation_time_seconds));
    result.response_path = detail::readString(is);
    result.error = detail::readString(is);
    return result;
  }
};

class FilePayloadTaskQueue {
 public:
  using Queue = boost::MemoryQueue<FilePayloadTask, 8192>;

  FilePayloadTaskQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  explicit FilePayloadTaskQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  void push(const FilePayloadTask& task) { queue_->push(task); }

  bool tryPop(FilePayloadTask& out) { return queue_->tryPop(out); }

  void receive(FilePayloadTask& out) { queue_->receive(out); }

  bool empty() const { return queue_->empty(); }

  void remove() { queue_->remove(); }

 private:
  std::unique_ptr<Queue> queue_;
};

class FilePayloadResultQueue {
 public:
  using Queue = boost::MemoryQueue<FilePayloadResult, 8192>;

  FilePayloadResultQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  explicit FilePayloadResultQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  bool push(const FilePayloadResult& result) {
    return queue_->tryPush(result);
  }

  bool blockingPop(FilePayloadResult& out) {
    queue_->receive(out);
    return !out.isDone();
  }

  void shutdown() { queue_->push(FilePayloadResult::done()); }

  void remove() { queue_->remove(); }

 private:
  std::unique_ptr<Queue> queue_;
};

class FilePayloadQueueSet {
 public:
  std::shared_ptr<FilePayloadTaskQueue> taskQueue;
  std::vector<std::shared_ptr<FilePayloadResultQueue>> resultQueues;

  explicit FilePayloadQueueSet(int numWorkers) {
    taskQueue = std::make_shared<FilePayloadTaskQueue>(
        tt::config::ttTaskQueueName(),
        static_cast<int>(tt::config::maxQueueSize()));
    resultQueues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; ++i) {
      resultQueues.emplace_back(std::make_shared<FilePayloadResultQueue>(
          std::string(tt::config::ttResultQueueName()) + std::to_string(i),
          static_cast<int>(tt::config::resultQueueCapacity())));
    }
  }

  ~FilePayloadQueueSet() { clear(); }

  void clear() {
    if (taskQueue) {
      taskQueue->remove();
    }
    for (auto& queue : resultQueues) {
      queue->shutdown();
      queue->remove();
    }
  }

  FilePayloadQueueSet(const FilePayloadQueueSet&) = delete;
  FilePayloadQueueSet& operator=(const FilePayloadQueueSet&) = delete;

  FilePayloadQueueSet(FilePayloadQueueSet&&) = default;
  FilePayloadQueueSet& operator=(FilePayloadQueueSet&&) = default;
};

}  // namespace tt::ipc::file_payload
