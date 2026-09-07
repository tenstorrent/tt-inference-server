// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "runtime/runners/i_embedding_runner.hpp"

namespace tt::runners {

/**
 * Embedding runner with no Python interpreter and no device.
 *
 * Answers with a unit-length vector of the configured dimension derived from a
 * hash of the input text, so the same input always yields the same vector and
 * different inputs yield different ones. That makes the service, batching,
 * codec, and HTTP layers testable on a machine with no Tenstorrent hardware,
 * and satisfies determinism assertions in the spec tests.
 *
 * The numbers are meaningless as embeddings - this is a plumbing stand-in, not
 * an approximation of the model.
 */
class MockEmbeddingRunner : public IEmbeddingRunner {
 public:
  explicit MockEmbeddingRunner(const config::EmbeddingConfig& config);

  bool warmup() override;

  std::vector<domain::EmbeddingResponse> run(
      const std::vector<domain::EmbeddingRequest>& requests) override;

  void close() override;

 private:
  config::EmbeddingConfig config_;
};

}  // namespace tt::runners
