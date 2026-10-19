// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>

#include "config/runner_config.hpp"
#include "ipc/embedding_ipc.hpp"
#include "runtime/runners/i_embedding_runner.hpp"
#include "runtime/runners/ipc_runner.hpp"

namespace tt::runners {

/**
 * Worker-side IPC loop for embedding models.
 *
 * Pulls EmbeddingIpcTask messages from the shared task queue, greedily drains
 * up to max_batch_size tasks when available, runs IEmbeddingRunner::run(batch),
 * and pushes per-request EmbeddingIpcResult messages to this worker's result
 * queue.
 */
class EmbeddingIpcRunner : public IRunner {
 public:
  EmbeddingIpcRunner(config::EmbeddingConfig config, int workerId);
  ~EmbeddingIpcRunner() override;

  bool warmup() override;
  void stop() override;
  const char* runnerType() const override { return "EmbeddingIpcRunner"; }

 protected:
  void run() override;

 private:
  config::EmbeddingConfig config;
  int workerIndex = 0;
  std::unique_ptr<IEmbeddingRunner> runner;
  std::unique_ptr<tt::ipc::embedding::EmbeddingTaskQueue> taskQueue;
  std::unique_ptr<tt::ipc::embedding::EmbeddingResultQueue> resultQueue;
  std::atomic<bool> stopped{false};
};

}  // namespace tt::runners
