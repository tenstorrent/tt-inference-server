// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>

#include "config/runner_config.hpp"
#include "runners/llm_runner/model_runner.hpp"

namespace tt::runners::llm_engine {

/**
 * IModelRunner that runs a HuggingFace causal LM (Qwen2.5-1.5B-Instruct by
 * default, configurable via HF_MODEL) via embedded Python interpreter
 * (pybind11). Calls tt_model_runners.hf_cpu_runner.HFCPURunner methods
 * directly in-process.
 *
 * This is a CPU-only runner (no tt-metal). KV cache is managed by the
 * Python-side DynamicCache; block_table is ignored.
 */
class QwenModelRunner : public IModelRunner {
 public:
  QwenModelRunner(const tt::config::LLMConfig& config, DecodeCallback callback);
  ~QwenModelRunner() override;
  void run(const std::vector<tt::domain::llm::Sequence*>& seqs,
           bool isPrefill) override;
  void exit() override;

  bool isReady() const { return initialized; }

 private:
  bool initialize();
  void failSequences(const std::vector<tt::domain::llm::Sequence*>& seqs);

  tt::config::LLMConfig config;
  DecodeCallback decodeCallback;
  std::atomic<bool> stop{false};
  bool initialized = false;
  bool lastStepWasPrefill = true;
};

std::unique_ptr<IModelRunner> makeQwenModelRunner(
    const tt::config::LLMConfig& config, DecodeCallback callback);

}  // namespace tt::runners::llm_engine
