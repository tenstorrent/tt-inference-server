// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/model_runner.hpp"

namespace llm_engine {

/**
 * IModelRunner that runs Llama-3.1-8B-Instruct via embedded Python interpreter (pybind11).
 * Calls tt_model_runners.llama_runner.Llama31_8BRunner methods directly in-process.
 */
class LlamaModelRunner : public IModelRunner {
 public:
  LlamaModelRunner(const Config& config, DecodeCallback callback);
  ~LlamaModelRunner() override;
  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override;
  void exit() override;

  bool is_ready() const { return initialized_; }

 private:
  bool initialize();
  void fail_sequences(const std::vector<Sequence*>& seqs);

  Config config_;
  DecodeCallback decode_callback_;
  std::atomic<bool> stop_{false};
  bool initialized_ = false;
};

std::unique_ptr<IModelRunner> make_llama_model_runner(const Config& config,
                                                      DecodeCallback callback);

}  // namespace llm_engine
