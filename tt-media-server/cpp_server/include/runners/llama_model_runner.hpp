// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
  void run(const std::vector<tt::domain::Sequence*>& seqs, bool isPrefill) override;
  void exit() override;

  bool isReady() const { return initialized; }

 private:
  bool initialize();
  void failSequences(const std::vector<tt::domain::Sequence*>& seqs);

  tt::config::LLMConfig config;
  DecodeCallback decodeCallback;
  std::atomic<bool> stop{false};
  bool initialized = false;
  bool lastStepWasPrefill = true;
};

std::unique_ptr<IModelRunner> makeLlamaModelRunner(
    const tt::config::LLMConfig& config, DecodeCallback callback);

}  // namespace tt::runners::llm_engine
