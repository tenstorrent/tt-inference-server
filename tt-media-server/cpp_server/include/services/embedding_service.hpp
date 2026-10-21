// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "ipc/embedding_ipc.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/request_pipeline.hpp"

namespace tt::services {

/**
 * Embedding request pipeline over WorkerManager + Boost shared-memory IPC
 * queues.
 */
class EmbeddingService
    : public BaseAsyncService<domain::EmbeddingRequest,
                              domain::EmbeddingResponse,
                              tt::ipc::embedding::EmbeddingQueueSet> {
 public:
  EmbeddingService();
  EmbeddingService(
      config::EmbeddingConfig config,
      std::unique_ptr<tt::worker::WorkerManager> workerManager,
      std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet> queueSet);
  ~EmbeddingService() override;

 protected:
  std::string validateRequest(
      const domain::EmbeddingRequest& request) const override;
  IpcTask buildIpcTask(const domain::EmbeddingRequest& request) override;
  domain::EmbeddingResponse buildResponse(uint32_t taskId,
                                          const IpcResult& result) override;
  void onBeforeStartWorkers(size_t numWorkers) override;
  void onTaskClaimed(size_t workerIdx, const IpcResult& result,
                     std::chrono::steady_clock::time_point claimTime,
                     double queueWaitMs, double shmEnqueueMs) override;
  void onTaskCompleted(size_t workerIdx, const IpcResult& result,
                       std::chrono::steady_clock::time_point gotResultTime,
                       std::chrono::steady_clock::time_point doneTime) override;
  void onStopped() override;

 private:
  struct WorkerBatchState {
    size_t batchSize = 0;
    double queueWaitMs = 0.0;
    double batchCollectMs = 0.0;
    double batchEnqueueMs = 0.0;
    double batchCompleteMs = 0.0;
    std::chrono::steady_clock::time_point claimTime{};
    std::chrono::steady_clock::time_point firstResultTime{};
  };

  struct DispatchSummary {
    struct Phase {
      const char* name;
      std::vector<double> samplesMs;
    };

    /** Emit a summary line every this many batches while serving, not only
     * at shutdown, so a CI log captured mid-run still shows the batch-size
     * histogram and per-phase timings. */
    static constexpr uint64_t kSummaryEveryBatches = 50;

    std::mutex mutex;
    Phase phases[5] = {{"queue_wait", {}},
                       {"batch_collect", {}},
                       {"shm_enqueue", {}},
                       {"shm_round_trip", {}},
                       {"completion", {}}};
    std::map<size_t, uint64_t> batchSizeCounts;
    uint64_t batchesSinceSummary = 0;

    /** Returns true when a periodic summary is due. */
    bool record(size_t batchSize, double queueWaitMs, double collectMs,
                double enqueueMs, double shmRoundTripMs, double completeMs);
    /** Logs and resets the window; `reason` tags the line ("periodic"/"final"). */
    void logSummary(const char* reason);
  };

  config::EmbeddingConfig config_;
  size_t maxInputBytes_ = 0;
  std::vector<WorkerBatchState> workerBatchStates_;
  DispatchSummary dispatchSummary_;
};

}  // namespace tt::services
