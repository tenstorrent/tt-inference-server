// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "runtime/runners/runner_base.hpp"

namespace tt::runners {

/**
 * What an embedding worker needs from a runner.
 * Derives from IRunnerBase so it can be registered and instantiated via
 * RunnerRegistry::createMedia<IEmbeddingRunner>.
 */
class IEmbeddingRunner : public IRunnerBase {
 public:
  ~IEmbeddingRunner() override = default;

  /** Bring up the model. False means the worker must exit. */
  bool warmup() override = 0;

  /** One forward pass. responses[i] answers requests[i], positionally. */
  virtual std::vector<domain::EmbeddingResponse> run(
      const std::vector<domain::EmbeddingRequest>& requests) = 0;

  /** Release model/device resources. Safe to call more than once. */
  virtual void close() = 0;

  void stop() override { close(); }
};

}  // namespace tt::runners
