// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>

#include "config/runner_config.hpp"
#include "runners/llm_runner/model_runner.hpp"

namespace tt::runners::llm_engine {

/**
 * IModelRunner that runs Llama-3.1-8B-Instruct via embedded Python interpreter
 * (pybind11). Calls tt_model_runners.llama_runner.Llama31_8BRunner methods
 * directly in-process.
 */
class LlamaModelRunner : public IModelRunner {
 public:
  LlamaModelRunner(const tt::config::LLMConfig& config,
                   DecodeCallback callback);
  ~LlamaModelRunner() override;
  void run(const std::vector<Sequence*>& seqs, bool isPrefill) override;
  void exit() override;

  bool isReady() const { return initialized_; }

 private:
  bool initialize();
  void failSequences(const std::vector<Sequence*>& seqs);

  tt::config::LLMConfig config_;
  DecodeCallback decode_callback_;
  std::atomic<bool> stop_{false};
  bool initialized_ = false;
  bool lastStepWasPrefill_ = true;
};

std::unique_ptr<IModelRunner> makeLlamaModelRunner(
    const tt::config::LLMConfig& config, DecodeCallback callback);

}  // namespace tt::runners::llm_engine
