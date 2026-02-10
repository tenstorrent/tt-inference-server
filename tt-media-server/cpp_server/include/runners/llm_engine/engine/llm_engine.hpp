#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/scheduler.hpp"
#include "llm_engine/sampling_params.hpp"

namespace llm_engine {

struct StepResult {
  std::vector<std::pair<int, std::vector<int64_t>>> outputs;
  int num_tokens;
};

class LLMEngine {
 public:
  explicit LLMEngine(const Config& config, std::unique_ptr<Scheduler> scheduler);
  ~LLMEngine();

  void add_request(std::vector<int64_t> prompt,
                   const SamplingParams& sampling_params = SamplingParams());
  StepResult step();
  bool is_finished() const;
  /// Each prompt uses the corresponding SamplingParams (same size as prompts).
  /// Empty or mismatched params: default SamplingParams for all.
  std::vector<std::vector<int64_t>> generate(
      const std::vector<std::vector<int64_t>>& prompts,
      const std::vector<SamplingParams>& sampling_params = {});
  /// Convenience: one max_tokens per prompt. Other sampling params use defaults.
  std::vector<std::vector<int64_t>> generate(
      const std::vector<std::vector<int64_t>>& prompts,
      const std::vector<int>& max_tokens_per_request);

 private:
  void exit();

  Config config_;
  std::unique_ptr<IModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  std::vector<std::unique_ptr<Sequence>> sequences_;
};

}  // namespace llm_engine
