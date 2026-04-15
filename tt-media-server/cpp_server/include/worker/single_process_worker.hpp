#pragma once

#include <unistd.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "config/runner_config.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/result_queue.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"

namespace tt::worker {

struct WorkerConfig {
  std::unordered_map<std::string, std::string> env_vars;
  std::shared_ptr<tt::runners::llm_engine::ITaskQueue> task_queue;
  std::shared_ptr<tt::ipc::IResultQueue> result_queue;
  std::shared_ptr<tt::ipc::ICancelQueue> cancel_queue;
  int worker_id;
  tt::config::RunnerConfig runner_config;
};

/**
 * Single process worker that runs an LLM engine.
 * Handles task processing and token generation.
 */
class SingleProcessWorker {
 public:
  SingleProcessWorker(WorkerConfig& cfg);
  ~SingleProcessWorker();

  void start();
  void stop();

  pid_t pid{-1};
  bool is_ready{false};
  bool is_alive{true};
  int worker_id{-1};
  WorkerConfig cfg;

 private:
  std::unique_ptr<tt::runners::IRunner> runner_;
};

}  // namespace tt::worker
