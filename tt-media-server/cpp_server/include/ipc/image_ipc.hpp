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

namespace tt::ipc::image {

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

struct ImageTask {
  uint32_t task_id = 0;
  uint32_t flags = 0;
  std::string request_path;
  std::string response_path;

  static constexpr uint32_t FLAG_DONE = 1;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }

  static ImageTask done() {
    ImageTask task;
    task.flags = FLAG_DONE;
    return task;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&task_id), sizeof(task_id));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    detail::writeString(os, request_path);
    detail::writeString(os, response_path);
  }

  static ImageTask deserialize(std::istream& is) {
    ImageTask task;
    is.read(reinterpret_cast<char*>(&task.task_id), sizeof(task.task_id));
    is.read(reinterpret_cast<char*>(&task.flags), sizeof(task.flags));
    task.request_path = detail::readString(is);
    task.response_path = detail::readString(is);
    return task;
  }
};

struct ImageResult {
  uint32_t task_id = 0;
  uint32_t flags = 0;
  double generation_time_seconds = 0.0;
  std::string response_path;
  std::string error;

  static constexpr uint32_t FLAG_DONE = 1;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }

  static ImageResult done() {
    ImageResult result;
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

  static ImageResult deserialize(std::istream& is) {
    ImageResult result;
    is.read(reinterpret_cast<char*>(&result.task_id), sizeof(result.task_id));
    is.read(reinterpret_cast<char*>(&result.flags), sizeof(result.flags));
    is.read(reinterpret_cast<char*>(&result.generation_time_seconds),
            sizeof(result.generation_time_seconds));
    result.response_path = detail::readString(is);
    result.error = detail::readString(is);
    return result;
  }
};

class ImageTaskQueue {
 public:
  using Queue = boost::MemoryQueue<ImageTask, 8192>;

  ImageTaskQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  explicit ImageTaskQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  void push(const ImageTask& task) { queue_->push(task); }

  bool tryPop(ImageTask& out) { return queue_->tryPop(out); }

  void receive(ImageTask& out) { queue_->receive(out); }

  bool empty() const { return queue_->empty(); }

  void remove() { queue_->remove(); }

 private:
  std::unique_ptr<Queue> queue_;
};

class ImageResultQueue {
 public:
  using Queue = boost::MemoryQueue<ImageResult, 8192>;

  ImageResultQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  explicit ImageResultQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  bool push(const ImageResult& result) { return queue_->tryPush(result); }

  bool blockingPop(ImageResult& out) {
    queue_->receive(out);
    return !out.isDone();
  }

  void shutdown() { queue_->push(ImageResult::done()); }

  void remove() { queue_->remove(); }

 private:
  std::unique_ptr<Queue> queue_;
};

class ImageQueueManager {
 public:
  std::shared_ptr<ImageTaskQueue> taskQueue;
  std::vector<std::shared_ptr<ImageResultQueue>> resultQueues;

  explicit ImageQueueManager(int numWorkers) {
    taskQueue = std::make_shared<ImageTaskQueue>(
        tt::config::ttTaskQueueName(), static_cast<int>(tt::config::maxQueueSize()));
    resultQueues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; ++i) {
      resultQueues.emplace_back(std::make_shared<ImageResultQueue>(
          std::string(tt::config::ttResultQueueName()) + std::to_string(i),
          static_cast<int>(tt::config::resultQueueCapacity())));
    }
  }

  ~ImageQueueManager() { clear(); }

  void clear() {
    if (taskQueue) {
      taskQueue->remove();
    }
    for (auto& queue : resultQueues) {
      queue->shutdown();
      queue->remove();
    }
  }

  ImageQueueManager(const ImageQueueManager&) = delete;
  ImageQueueManager& operator=(const ImageQueueManager&) = delete;

  ImageQueueManager(ImageQueueManager&&) = default;
  ImageQueueManager& operator=(ImageQueueManager&&) = default;
};

}  // namespace tt::ipc::image
