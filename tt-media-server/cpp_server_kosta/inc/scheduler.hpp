#pragma once
#include "domain.hpp"
#include <boost/interprocess/ipc/message_queue.hpp>
#include <sys/types.h>
#include <thread>
#include <vector>
#include <string>
#include <memory>
#include <atomic>

using namespace boost::interprocess;

class Scheduler {
public:
  static Scheduler &getInstace();
  void put(const domain::Request &req);
  void start(int numOfWorkers, const std::string &model_path);
  void stop();
  static void cleanup();  // Static for signal handler
  
  // Metrics
  int getNumWorkers() const { return numWorkers; }
  int getNumTaskQueues() const { return numTaskQueues; }
  size_t getQueueSize() const;
  size_t getQueueCapacity() const;

private:
  Scheduler();
  int numWorkers = 0;
  int numTaskQueues = 0;
  std::string modelPath;
  
  // Phase 1: Sharded task queues for reduced lock contention
  std::vector<std::unique_ptr<message_queue>> taskQueues;
  std::atomic<uint64_t> roundRobinCounter{0};
  
  std::vector<std::thread *> monitors;
  std::vector<message_queue *> resultQueues;
  std::vector<pid_t> workerPids;
  void monitorLoop(int workerId);
  void spawnWorker(int workerId);
};
