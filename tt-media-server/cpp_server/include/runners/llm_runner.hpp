#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "config/runner_config.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/result_queue.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "ipc/task_queue.hpp"
#include "runners/runner_interface.hpp"

namespace tt::services {
class MemoryManager;
}  // namespace tt::services

namespace tt::runners {

class GuidedDecoderManager;
using namespace tt::runners::llm_engine;

class LLMRunner : public IRunner {
 public:
  LLMRunner(const config::LLMConfig& config, ipc::IResultQueue* resultQueue,
            ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue = nullptr);
  ~LLMRunner() override;

  Scheduler& getScheduler() { return *scheduler; }

  void run() override;
  void stop() override;
  const char* runnerType() const override { return "LLMRunner"; }

 private:
  void step();
  void memoryLoop();
  void exit();
  void applyGuidedDecodingMasks(const std::vector<tt::domain::Sequence*>& seqs,
                                bool isPrefill);

  config::LLMConfig config;
  ipc::IResultQueue* resultQueue;
  ipc::ICancelQueue* cancelQueue;  // nullable; owned by caller
  std::unique_ptr<IModelRunner> modelRunner;
  std::unique_ptr<Scheduler> scheduler;
  std::atomic<bool> stopped{false};

  std::unique_ptr<tt::services::MemoryManager> memoryManager;
  std::thread memoryThread;

  std::unique_ptr<GuidedDecoderManager> guidedDecoder;
};

}  // namespace tt::runners
