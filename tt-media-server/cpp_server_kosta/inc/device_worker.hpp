#pragma once
#include "boost/interprocess/ipc/message_queue.hpp"
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include "runner.hpp"
#include <fmt/core.h>
#include <string>
#include <vector>

using namespace boost::interprocess;

class DeviceWorker {
private:
  int workerId;
  int queueIndex;  // Phase 1: Assigned task queue index
  ModelRunner runner;
  message_queue taskQueue;
  message_queue responseQueue;
  size_t maxBatchSize;
  boost::asio::thread_pool pool;

public:
  // Phase 1: Accept queue index for sharded task queues
  // By default, queueIndex = workerId (one queue per worker)
  DeviceWorker(int id, const std::string &model_path, size_t batchSize = 32, int queueIdx = -1)
      : workerId(id),
        queueIndex(queueIdx >= 0 ? queueIdx : id),  // Default: queue index = worker id
        runner(model_path),
        maxBatchSize(batchSize),
        taskQueue(open_only, fmt::format("task_queue{}", queueIndex).c_str()),
        responseQueue(open_only, fmt::format("result_queue{}", id).c_str()),
        pool(batchSize) {}
  void start();
};
