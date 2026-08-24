// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "runtime/runners/i_embedding_runner.hpp"

namespace tt::runners {

namespace detail {
// Template-method base for the per-model implementations (defined in
// embedding_runner.cpp). It owns the shared pipeline - device open,
// tokenizer, warmup, tokenize->forward->extract, close - and each model
// subclass overrides only the steps that differ: which tt-metal module and
// class to load, the constructor kwargs, and how to pull the dense vectors
// out of forward()'s result. Kept behind this forward declaration so pybind11
// types never leak into headers.
struct EmbeddingImpl;
}  // namespace detail

/**
 * Embedding runner that drives tt-metal directly.
 *
 * Uses pybind11 (embedded interpreter) to import ttnn and the model's
 * generator class from tt-metal's models.demos - there is no tt-media-server
 * Python layer involved. Tokenization goes through the model's HuggingFace
 * AutoTokenizer for exact parity with the Python server. Python errors are
 * captured with full tracebacks and surfaced as per-request error responses
 * rather than swallowed.
 */
class EmbeddingRunner : public IEmbeddingRunner {
 public:
  explicit EmbeddingRunner(const config::EmbeddingConfig& config);
  ~EmbeddingRunner() override;

  // Prevent copying
  EmbeddingRunner(const EmbeddingRunner&) = delete;
  EmbeddingRunner& operator=(const EmbeddingRunner&) = delete;

  /** Open the ttnn mesh device, load tokenizer and model weights, and run
   * one warmup forward pass. */
  bool warmup() override;

  std::vector<domain::EmbeddingResponse> run(
      const std::vector<domain::EmbeddingRequest>& requests) override;

  /** Close the mesh device and drop the Python objects. The interpreter
   * itself is left running. */
  void close() override;

 private:
  config::EmbeddingConfig config_;
  std::unique_ptr<detail::EmbeddingImpl> impl_;
};

}  // namespace tt::runners
