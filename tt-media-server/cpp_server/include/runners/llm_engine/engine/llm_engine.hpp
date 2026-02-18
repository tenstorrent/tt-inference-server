#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/scheduler.hpp"

namespace llm_engine {

using TokenCallback =
    std::function<void(int seq_id, int64_t token_id, bool finished)>;

class LLMEngine {
 public:
  LLMEngine(const Config& config, TokenCallback on_token, std::unique_ptr<Scheduler> scheduler);
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
