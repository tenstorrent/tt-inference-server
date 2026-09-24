// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

namespace tt::runners {

/**
 * A batch whose host-side stage (validation, tokenization) already ran.
 * Produced by prepare() on one thread and consumed by runPrepared() on
 * another, so the worker can tokenize batch N+1 while batch N occupies the
 * device (double buffering).
 */
struct PreparedBatch {
  std::vector<domain::EmbeddingRequest> requests;

  /// Non-empty when preparation already produced the final answers (e.g.
  /// validation or tokenization failed); runPrepared() then returns these.
  std::vector<domain::EmbeddingResponse> immediate;

  /// Runner-specific host state (e.g. tokenized input tensors).
  std::shared_ptr<void> payload;
};

/**
 * What an embedding worker needs from a runner, and nothing more.
 * EmbeddingService owns the loop and drives the runner directly with these
 * calls.
 */
class IEmbeddingRunner {
 public:
  virtual ~IEmbeddingRunner() = default;

  /** Bring up the model. False means the worker must exit. */
  virtual bool warmup() = 0;

  /** One forward pass. responses[i] answers requests[i], positionally. */
  virtual std::vector<domain::EmbeddingResponse> run(
      const std::vector<domain::EmbeddingRequest>& requests) = 0;

  /** Host-side stage of a forward pass. Must be safe to run concurrently
   * with runPrepared() of an earlier batch. */
  virtual PreparedBatch prepare(
      std::vector<domain::EmbeddingRequest> requests) {
    return PreparedBatch{std::move(requests), {}, nullptr};
  }

  /** Device stage of a forward pass. responses[i] answers
   * batch.requests[i], positionally. */
  virtual std::vector<domain::EmbeddingResponse> runPrepared(
      PreparedBatch& batch) {
    if (!batch.immediate.empty()) return std::move(batch.immediate);
    return run(batch.requests);
  }

  /** Release model/device resources. Safe to call more than once. */
  virtual void close() = 0;
};

/** Build the runner named by cfg.runner_type. */
std::unique_ptr<IEmbeddingRunner> makeEmbeddingRunner(
    const config::EmbeddingConfig& cfg);

}  // namespace tt::runners
