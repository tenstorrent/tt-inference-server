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
 * What an embedding worker needs from a runner, and nothing more.
 *
 * Deliberately not IRunner: that interface is for loop-driven IPC workers and
 * forces a no-arg run() this path never uses. EmbeddingService owns the loop
 * and drives the runner directly with these three calls.
 */
class IEmbeddingRunner {
 public:
  virtual ~IEmbeddingRunner() = default;

  /** Bring up the model. False means the worker must exit. */
  virtual bool warmup() = 0;

  /** One forward pass. responses[i] answers requests[i], positionally. */
  virtual std::vector<domain::EmbeddingResponse> run(
      const std::vector<domain::EmbeddingRequest>& requests) = 0;

  /** Release model/device resources. Safe to call more than once. */
  virtual void close() = 0;
};

/** Build the runner named by cfg.runner_type. EMBEDDING_MOCK yields a runner
 *  that needs neither Python nor a device. */
std::unique_ptr<IEmbeddingRunner> makeEmbeddingRunner(
    const config::EmbeddingConfig& cfg);

}  // namespace tt::runners
