#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include "runners/runner_interface.hpp"
#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "ipc/shared_memory.hpp"
#include "ipc/cancel_queue.hpp"

namespace tt::runners {
  using namespace llm_engine;

class LLMRunner : public IRunner {
 public:
  LLMRunner(const Config& config, ipc::TokenRingBuffer<65536>* result_queue,
            ITaskQueue* task_queue, ipc::CancelQueue* cancel_queue = nullptr);
  ~LLMRunner() override;

  Scheduler& scheduler() { return *scheduler_; }

  void run() override;
  void stop() override;
  const char* runner_type() const override { return "LLMRunner"; }

 private:
  void step();
  void exit();

  Config config_;
  ipc::TokenRingBuffer<65536>* result_queue_;
  ipc::CancelQueue* cancel_queue_;
  std::unique_ptr<IModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  DecodeQueue decode_queue_;
  std::atomic<bool> stopped_{false};
};

}  // namespace tt::runners
