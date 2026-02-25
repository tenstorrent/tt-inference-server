#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

#include "runners/runner_interface.hpp"
#include "runners/runner_result.hpp"
#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/task_queue.hpp"

namespace tt::runners {
  using namespace llm_engine;

class LLMRunner : public IRunner {
 public:
  LLMRunner(const Config& config, ResultCallback on_result, ITaskQueue* task_queue);
  ~LLMRunner() override;

  Scheduler& scheduler() { return *scheduler_; }

  void run() override;
  void stop() override;
  const char* runner_type() const override { return "LLMRunner"; }

 private:
  void step();
  void drain_decode_results();
  void exit();

  Config config_;
  ResultCallback on_result_;
  std::unique_ptr<IModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  DecodeQueue decode_queue_;
  std::atomic<bool> stopped_{false};
};

}  // namespace llm_engine
