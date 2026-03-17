#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "ipc/shared_memory.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"

namespace tt::runners {
using namespace llm_engine;

class LLMRunner : public IRunner {
 public:
  LLMRunner(const config::LLMConfig& config,
            ipc::TokenRingBuffer<65536>* resultQueue, ITaskQueue* taskQueue);
  ~LLMRunner() override;

  Scheduler& getScheduler() { return *scheduler; }

  void run() override;
  void stop() override;
  const char* runner_type() const override { return "LLMRunner"; }

 private:
  void step();
  void exit();

  config::LLMConfig config;
  ipc::TokenRingBuffer<65536>* resultQueue;
  std::unique_ptr<IModelRunner> modelRunner;
  std::unique_ptr<Scheduler> scheduler;
  DecodeQueue decodeQueue;
  std::atomic<bool> stopped{false};
};

}  // namespace tt::runners
