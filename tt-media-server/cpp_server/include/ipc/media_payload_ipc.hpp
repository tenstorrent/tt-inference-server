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

namespace tt::ipc::media_payload {

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

struct MediaPayloadTask {
  uint32_t task_id = 0;
  uint32_t flags = 0;
  std::string request_path;
  std::string response_path;

  static constexpr uint32_t FLAG_DONE = 1;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }

  static MediaPayloadTask done() {
    MediaPayloadTask task;
    task.flags = FLAG_DONE;
    return task;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&task_id), sizeof(task_id));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    detail::writeString(os, request_path);
    detail::writeString(os, response_path);
  }

  static MediaPayloadTask deserialize(std::istream& is) {
    MediaPayloadTask task;
    is.read(reinterpret_cast<char*>(&task.task_id), sizeof(task.task_id));
    is.read(reinterpret_cast<char*>(&task.flags), sizeof(task.flags));
    task.request_path = detail::readString(is);
    task.response_path = detail::readString(is);
    return task;
  }
};

struct MediaPayloadResult {
  uint32_t task_id = 0;
  uint32_t flags = 0;
  double generation_time_seconds = 0.0;
  std::string response_path;
  std::string error;

  static constexpr uint32_t FLAG_DONE = 1;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }

  static MediaPayloadResult done() {
    MediaPayloadResult result;
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

  static MediaPayloadResult deserialize(std::istream& is) {
    MediaPayloadResult result;
    is.read(reinterpret_cast<char*>(&result.task_id), sizeof(result.task_id));
    is.read(reinterpret_cast<char*>(&result.flags), sizeof(result.flags));
    is.read(reinterpret_cast<char*>(&result.generation_time_seconds),
            sizeof(result.generation_time_seconds));
    result.response_path = detail::readString(is);
    result.error = detail::readString(is);
    return result;
  }
};

class MediaPayloadTaskQueue {
 public:
  using Queue = boost::MemoryQueue<MediaPayloadTask, 8192>;

  MediaPayloadTaskQueue(const std::string& name, int capacity)
      : queue(std::make_unique<Queue>(name, capacity)) {}

  explicit MediaPayloadTaskQueue(const std::string& name)
      : queue(Queue::openExisting(name)) {}

  void push(const MediaPayloadTask& task) { queue->push(task); }

  bool tryPop(MediaPayloadTask& out) { return queue->tryPop(out); }

  void receive(MediaPayloadTask& out) { queue->receive(out); }

  bool empty() const { return queue->empty(); }

  void remove() { queue->remove(); }

 private:
  std::unique_ptr<Queue> queue;
};

class MediaPayloadResultQueue {
 public:
  using Queue = boost::MemoryQueue<MediaPayloadResult, 8192>;

  MediaPayloadResultQueue(const std::string& name, int capacity)
      : queue(std::make_unique<Queue>(name, capacity)) {}

  explicit MediaPayloadResultQueue(const std::string& name)
      : queue(Queue::openExisting(name)) {}

  bool push(const MediaPayloadResult& result) { return queue->tryPush(result); }

  bool blockingPop(MediaPayloadResult& out) {
    queue->receive(out);
    return !out.isDone();
  }

  void shutdown() { queue->push(MediaPayloadResult::done()); }

  void remove() { queue->remove(); }

 private:
  std::unique_ptr<Queue> queue;
};

class MediaPayloadQueueSet {
 public:
  std::shared_ptr<MediaPayloadTaskQueue> taskQueue;
  std::vector<std::shared_ptr<MediaPayloadResultQueue>> resultQueues;

  explicit MediaPayloadQueueSet(int numWorkers) {
    taskQueue = std::make_shared<MediaPayloadTaskQueue>(
        tt::config::ttMediaTaskQueueName(),
        static_cast<int>(tt::config::maxQueueSize()));
    resultQueues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; ++i) {
      resultQueues.emplace_back(std::make_shared<MediaPayloadResultQueue>(
          std::string(tt::config::ttMediaResultQueueName()) + std::to_string(i),
          static_cast<int>(tt::config::resultQueueCapacity())));
    }
  }

  ~MediaPayloadQueueSet() { clear(); }

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

  MediaPayloadQueueSet(const MediaPayloadQueueSet&) = delete;
  MediaPayloadQueueSet& operator=(const MediaPayloadQueueSet&) = delete;

  MediaPayloadQueueSet(MediaPayloadQueueSet&&) = default;
  MediaPayloadQueueSet& operator=(MediaPayloadQueueSet&&) = default;
};

}  // namespace tt::ipc::media_payload
