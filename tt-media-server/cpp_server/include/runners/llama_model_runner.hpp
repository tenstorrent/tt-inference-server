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
 * 
 * Thread Safety:
 * - Uses Python's GIL (Global Interpreter Lock) for thread-safe Python calls
 * - All Python interactions are wrapped in py::gil_scoped_acquire
 * - PyEval_SaveThread() releases GIL when not executing Python code
 * - The Python interpreter is a single global resource, so all inference through
 *   this runner is serialized at the Python boundary
 * 
 * Performance Considerations:
 * - GIL serialization may become a bottleneck under high concurrency
 * - Consider multiple worker processes if GIL contention impacts throughput
 * 
 * Resource Management:
 * - Python interpreter initialization is handled in initialize()
 * - Global Python objects (g_runner, g_step_seq_class) cleaned up in exit()
 * - Destructor ensures exit() is called for proper cleanup
 * - If Python interpreter was initialized by this class, it is NOT finalized
 *   to avoid issues with other potential Python code in the process
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
  bool owns_interpreter_ = false;
};

std::unique_ptr<IModelRunner> make_llama_model_runner(const Config& config,
                                                      DecodeCallback callback);

}  // namespace llm_engine
