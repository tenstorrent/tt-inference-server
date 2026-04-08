#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "config/runner_config.hpp"
#include "ipc/boost_ipc_queue.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/token_ring_buffer.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/runner_interface.hpp"

namespace tt::services {
class MemoryManager;
}

namespace tt::runners {
using namespace tt::runners::llm_engine;

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
  void memoryLoop();
  void exit();

  config::LLMConfig config_;
  ipc::TokenRingBuffer<65536>* result_queue_;
  ipc::ICancelQueue* cancel_queue_;  // nullable; owned by caller
  std::unique_ptr<IModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  std::atomic<bool> stopped_{false};

  std::unique_ptr<tt::services::MemoryManager> memoryManager;
  std::thread memoryThread;
};

}  // namespace tt::runners
