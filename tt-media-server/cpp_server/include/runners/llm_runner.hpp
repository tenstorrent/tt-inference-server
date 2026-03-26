#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "ipc/cancel_queue.hpp"
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
            ipc::TokenRingBuffer<65536>* resultQueue, ITaskQueue* taskQueue,
            ipc::ICancelQueue* cancelQueue = nullptr);
  ~LLMRunner() override;

  Scheduler& scheduler() { return *scheduler_; }

  void run() override;
  void stop() override;
  const char* runnerType() const { return "LLMRunner"; }

 private:
  void step();
  void exit();

  config::LLMConfig config_;
  ipc::TokenRingBuffer<65536>* result_queue_;
  ipc::ICancelQueue* cancel_queue_;  // nullable; owned by caller
  std::unique_ptr<IModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  std::atomic<bool> stopped_{false};
};

}  // namespace tt::runners
