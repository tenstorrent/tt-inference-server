// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/model_runner.hpp"

namespace llm_engine {

/** IModelRunner that runs Llama-3.1-8B via Python subprocess (tt_model_runners.llama_runner). */
class PipeLlamaModelRunner : public IModelRunner {
 public:
  PipeLlamaModelRunner(const Config& config, DecodeCallback callback);
  ~PipeLlamaModelRunner() override;
  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override;
  void exit() override;

  /** True if Python subprocess was spawned successfully. */
  bool is_spawned() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** Create pipe runner when MODEL_RUNNER=ttnn_test. Returns nullptr if spawn fails. */
std::unique_ptr<IModelRunner> make_pipe_llama_model_runner(const Config& config,
                                                           DecodeCallback callback);

}  // namespace llm_engine
