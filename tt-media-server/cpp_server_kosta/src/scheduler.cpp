#include "scheduler.hpp"
#include "base_service.hpp"
#include "device_worker.hpp"
#include <chrono>
#include <fmt/format.h>
#include <json/json.h>
#include <mutex>
#include <random>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

Scheduler &Scheduler::getInstace() {
  static Scheduler instance;
  return instance;
}

Scheduler::Scheduler() {
  // Task queues are created in start() when we know numWorkers
}

void Scheduler::put(const domain::Request &req) {
  // Round-robin distribution across sharded task queues
  uint64_t idx = roundRobinCounter.fetch_add(1, std::memory_order_relaxed);
  int queueIdx = static_cast<int>(idx % numTaskQueues);
  taskQueues[queueIdx]->send(&req, sizeof(req), 0);
}

static std::string generateCompletionId() {
  static std::mutex rng_mutex;
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static const char* hex = "0123456789abcdef";
  
  std::string id = "cmpl-";
  {
    std::lock_guard<std::mutex> lock(rng_mutex);
    for (int i = 0; i < 24; ++i) {
      id += hex[dis(gen)];
    }
  }
  return id;
}

void Scheduler::monitorLoop(int workerId) {
  Json::StreamWriterBuilder writerBuilder;
  writerBuilder["indentation"] = "";  // No pretty printing for SSE
  
  while (true) {
    // Phase 2: Receive batched responses instead of individual tokens
    domain::BatchedResponse batch;
    unsigned int priority;
    message_queue::size_type recvd_size;

    resultQueues[workerId]->receive(&batch, sizeof(batch), recvd_size, priority);

    auto &tracker = Tracker::getInstance();
    PendingRequest *pending = tracker.get(batch.request_id);
    if (pending && pending->stream) {
      // Process all tokens in the batch
      for (int i = 0; i < batch.num_tokens; i++) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();

        Json::Value response;
        response["id"] = generateCompletionId();
        response["object"] = "text_completion";
        response["created"] = static_cast<Json::Int64>(timestamp);
        response["model"] = "some-model";
        
        // choices array - use token from batch
        response["choices"][0]["text"] = std::string(batch.tokens[i]);
        response["choices"][0]["index"] = 0;
        response["choices"][0]["logprobs"] = Json::Value::null;
        
        // Only the last token in a finished batch gets finish_reason
        bool isLastToken = batch.is_finished && (i == batch.num_tokens - 1);
        if (isLastToken) {
          response["choices"][0]["finish_reason"] = "stop";
        } else {
          response["choices"][0]["finish_reason"] = Json::Value::null;
        }
        
        // usage (approximate - actual token counting would require a tokenizer)
        response["usage"]["prompt_tokens"] = 0;
        response["usage"]["completion_tokens"] = 0;
        response["usage"]["total_tokens"] = 0;

        // Write as SSE format
        std::string jsonStr = Json::writeString(writerBuilder, response);
        std::string output = "data: " + jsonStr + "\n\n";
        
        // Drogon's ResponseStream is thread-safe, can call send() directly
        pending->stream->send(output);

        if (isLastToken) {
          // Send SSE termination signal and close stream
          pending->stream->send("data: [DONE]\n\n");
          pending->stream->close();
          tracker.remove(batch.request_id);
        }
      }
    }
  }
}

void Scheduler::spawnWorker(int workerId) {
  pid_t pid = fork();
  if (pid == 0) {
    DeviceWorker worker(workerId, modelPath);
    worker.start();
    _exit(0);
  } else if (pid > 0) {
    workerPids.push_back(pid);
  } else {
    throw std::runtime_error("Failed to fork worker process");
  }
}

void Scheduler::start(int numOfWorkers, const std::string &model_path) {
  this->numWorkers = numOfWorkers;
  this->modelPath = model_path;
  this->resultQueues = std::vector<message_queue *>(numOfWorkers);
  this->monitors = std::vector<std::thread *>(numOfWorkers);
  this->workerPids.reserve(numOfWorkers);
  
  // Phase 1: Create sharded task queues (one per worker for zero contention)
  this->numTaskQueues = numOfWorkers;
  this->taskQueues.reserve(numTaskQueues);
  
  for (int i = 0; i < numTaskQueues; i++) {
    std::string queueName = fmt::format("task_queue{}", i);
    message_queue::remove(queueName.c_str());
    // Each queue needs enough capacity to handle burst traffic
    // 256 per queue allows handling 8192 total concurrent requests
    size_t queueCapacity = 256;
    taskQueues.push_back(std::make_unique<message_queue>(
        create_only, queueName.c_str(), queueCapacity, sizeof(domain::Request)));
  }

  for (int i = 0; i < numOfWorkers; i++) {
    std::string queueName = fmt::format("result_queue{}", i);
    message_queue::remove(queueName.c_str());
    resultQueues[i] =
        new message_queue(create_only, queueName.c_str(),
                          1024, sizeof(domain::BatchedResponse));
    spawnWorker(i);
    monitors[i] = new std::thread(&Scheduler::monitorLoop, this, i);
  }
}

void Scheduler::stop() {
  for (pid_t pid : workerPids) {
    kill(pid, SIGTERM);
  }

  for (pid_t pid : workerPids) {
    waitpid(pid, nullptr, 0);
  }
  workerPids.clear();

  // Cleanup task queues
  for (int i = 0; i < numTaskQueues; i++) {
    message_queue::remove(fmt::format("task_queue{}", i).c_str());
  }
  taskQueues.clear();

  for (int i = 0; i < numWorkers; i++) {
    delete resultQueues[i];
    message_queue::remove(fmt::format("result_queue{}", i).c_str());
  }
  resultQueues.clear();
}

void Scheduler::cleanup() {
  Scheduler &instance = getInstace();
  // Remove all task queues
  for (int i = 0; i < instance.numTaskQueues; i++) {
    message_queue::remove(fmt::format("task_queue{}", i).c_str());
  }
  instance.stop();
}

size_t Scheduler::getQueueSize() const {
  size_t total = 0;
  for (const auto& queue : taskQueues) {
    if (queue) total += queue->get_num_msg();
  }
  return total;
}

size_t Scheduler::getQueueCapacity() const {
  size_t total = 0;
  for (const auto& queue : taskQueues) {
    if (queue) total += queue->get_max_msg();
  }
  return total;
}
