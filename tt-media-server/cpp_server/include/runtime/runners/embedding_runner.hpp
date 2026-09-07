// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "runtime/runners/i_embedding_runner.hpp"

namespace tt::runners {

/**
 * Embedding runner that calls a Python model runner in-process.
 *
 * Uses pybind11 (embedded interpreter). The runner class is resolved by
 * Python's tt_model_runners/runner_fabric.py from the MODEL_RUNNER env var
 * the worker exports. Python errors are captured with full tracebacks and
 * surfaced as per-request error responses rather than swallowed.
 */
class EmbeddingRunner : public IEmbeddingRunner {
 public:
  explicit EmbeddingRunner(const config::EmbeddingConfig& config);
  ~EmbeddingRunner() override;

  // Prevent copying
  EmbeddingRunner(const EmbeddingRunner&) = delete;
  EmbeddingRunner& operator=(const EmbeddingRunner&) = delete;

  /** Import the Python module, construct the runner, initialize the TTNN
   * device, and run the model's warmup. */
  bool warmup() override;

  std::vector<domain::EmbeddingResponse> run(
      const std::vector<domain::EmbeddingRequest>& requests) override;

  /** Drop the Python objects. The interpreter itself is left running. */
  void close() override;

 private:
  config::EmbeddingConfig config_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::runners
