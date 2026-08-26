// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <memory>
#include <stdexcept>

#include "config/types.hpp"
#include "runtime/runners/embedding_runner.hpp"
#include "runtime/runners/i_embedding_runner.hpp"
#include "runtime/runners/mock_embedding_runner.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

std::unique_ptr<IEmbeddingRunner> makeEmbeddingRunner(
    const config::EmbeddingConfig& cfg) {
  TT_LOG_INFO("[EmbeddingRunner] Building runner_type={}",
              config::toString(cfg.runner_type));

  switch (cfg.runner_type) {
    case config::ModelRunnerType::EMBEDDING_MOCK:
      return std::make_unique<MockEmbeddingRunner>(cfg);
    case config::ModelRunnerType::TT_BGE_LARGE_EN:
    case config::ModelRunnerType::TT_BGE_M3:
    case config::ModelRunnerType::TT_QWEN_EMBEDDING_8B:
      return std::make_unique<EmbeddingRunner>(cfg);
    default:
      throw std::runtime_error(
          "[EmbeddingRunner] runner_type=" + config::toString(cfg.runner_type) +
          " is not an embedding runner");
  }
}

}  // namespace tt::runners
