#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

#include "runners/llm_engine/config.hpp"
#include "runners/llm_engine/engine/model_runner.hpp"
#include "runners/llm_engine/engine/scheduler.hpp"

namespace llm_engine {

using TokenCallback =
    std::function<void(TaskID task_id, uint64_t token_id, bool finished)>;

/** Factory: (Config, DecodeCallback) -> unique_ptr<IModelRunner>. If null, engine uses make_model_runner. */
using ModelRunnerFactory = std::function<std::unique_ptr<IModelRunner>(const Config&, DecodeCallback)>;

class LLMEngine {
 public:
  /** When factory is null, uses make_model_runner(config, callback). */
  LLMEngine(const Config& config, TokenCallback on_token, std::unique_ptr<Scheduler> scheduler,
            ModelRunnerFactory factory = nullptr);
  ~LLMEngine();

  Scheduler& scheduler() { return *scheduler_; }

  void run();
  void stop();

 private:
  void step();
  void drain_decode_results();
  void exit();

  Config config_;
  TokenCallback on_token_;
  std::unique_ptr<IModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  DecodeQueue decode_queue_;
  std::atomic<bool> stopped_{false};
};

}  // namespace llm_engine
