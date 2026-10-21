// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "services/embedding_service.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {

double meanOf(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double sum = 0.0;
  for (double x : v) sum += x;
  return sum / static_cast<double>(v.size());
}

double p99Of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t idx =
      (v.size() * 99 + 99) / 100 == 0 ? 0 : (v.size() * 99 + 99) / 100 - 1;
  return v[std::min(idx, v.size() - 1)];
}

std::unique_ptr<tt::worker::WorkerManager> makeDefaultEmbeddingWorkerManager() {
  tt::worker::WorkerManagerOptions opts;
  opts.sequentialLeadWorkerWarmup = true;
  opts.warmupTimeoutMs = tt::config::embeddingWarmupTimeoutMs();
  return std::make_unique<tt::worker::WorkerManager>(tt::config::numWorkers(),
                                                     opts);
}

std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet>
makeDefaultEmbeddingQueueSet(const config::EmbeddingConfig& cfg) {
  return std::make_unique<tt::ipc::embedding::EmbeddingQueueSet>(
      static_cast<int>(tt::config::numWorkers()), cfg.max_seq_len,
      cfg.max_batch_size);
}

}  // namespace

bool EmbeddingService::DispatchSummary::record(
    size_t batchSize, double queueWaitMs, double collectMs, double enqueueMs,
    double shmRoundTripMs, double completeMs) {
  std::lock_guard<std::mutex> lock(mutex);
  phases[0].samplesMs.push_back(queueWaitMs);
  phases[1].samplesMs.push_back(collectMs);
  phases[2].samplesMs.push_back(enqueueMs);
  phases[3].samplesMs.push_back(shmRoundTripMs);
  phases[4].samplesMs.push_back(completeMs);
  batchSizeCounts[batchSize] += 1;
  return ++batchesSinceSummary >= kSummaryEveryBatches;
}

void EmbeddingService::DispatchSummary::logSummary(const char* reason) {
  std::lock_guard<std::mutex> lock(mutex);
  if (batchSizeCounts.empty()) return;

  std::ostringstream line;
  uint64_t batches = 0;
  uint64_t requests = 0;
  for (const auto& [size, count] : batchSizeCounts) {
    batches += count;
    requests += count * size;
  }
  line << "[EmbeddingService] Dispatch summary (" << reason
       << "): batches=" << batches << " requests=" << requests;
  for (auto& phase : phases) {
    line << " " << phase.name << "_ms=" << meanOf(phase.samplesMs) << "/"
         << p99Of(phase.samplesMs) << "(mean/p99)";
  }
  line << " batches:";
  for (auto it = batchSizeCounts.rbegin(); it != batchSizeCounts.rend(); ++it) {
    line << " " << it->second << "x" << it->first;
  }
  TT_LOG_INFO("{}", line.str());
  batchSizeCounts.clear();
  for (auto& phase : phases) {
    phase.samplesMs.clear();
  }
  batchesSinceSummary = 0;
}

EmbeddingService::EmbeddingService()
    : EmbeddingService(
          config::embeddingEngineConfig(), makeDefaultEmbeddingWorkerManager(),
          makeDefaultEmbeddingQueueSet(config::embeddingEngineConfig())) {}

EmbeddingService::EmbeddingService(
    config::EmbeddingConfig config,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet> queueSet)
    : BaseAsyncService("EmbeddingService", std::move(workerManager),
                       std::move(queueSet)),
      config_(std::move(config)),
      maxInputBytes_(tt::ipc::embedding::maxInputBytesForTaskMsg(
          this->queueSet()
              ? this->queueSet()->maxTaskMsgBytes
              : tt::ipc::embedding::EMBEDDING_MAX_TASK_MSG_BYTES)) {
  TT_LOG_INFO(
      "[EmbeddingService] Initialized with {} workers, max_batch_size={}, "
      "max_queue_size={}, max_input_bytes={}",
      numWorkers(), config_.max_batch_size, this->maxQueueSize, maxInputBytes_);
}

EmbeddingService::~EmbeddingService() { stop(); }

std::string EmbeddingService::validateRequest(
    const domain::EmbeddingRequest& request) const {
  if (request.input.size() > maxInputBytes_) {
    return "Input length (" + std::to_string(request.input.size()) +
           " bytes) exceeds maximum allowed (" +
           std::to_string(maxInputBytes_) + " bytes)";
  }
  return {};
}

EmbeddingService::IpcTask EmbeddingService::buildIpcTask(
    const domain::EmbeddingRequest& request) {
  return tt::ipc::embedding::EmbeddingIpcTask::fromDomain(request);
}

domain::EmbeddingResponse EmbeddingService::buildResponse(
    uint32_t /*taskId*/, const IpcResult& result) {
  return result.toDomain();
}

void EmbeddingService::onBeforeStartWorkers(size_t numWorkers) {
  workerBatchStates_.assign(numWorkers, WorkerBatchState{});
}

void EmbeddingService::onTaskClaimed(
    size_t workerIdx, const IpcResult& result,
    std::chrono::steady_clock::time_point claimTime, double queueWaitMs,
    double shmEnqueueMs) {
  if (workerIdx >= workerBatchStates_.size()) return;
  auto& state = workerBatchStates_[workerIdx];
  if (result.totalTokens > 0) {
    state.batchSize = static_cast<size_t>(result.totalTokens);
    state.queueWaitMs = queueWaitMs;
    state.batchCollectMs = static_cast<double>(result.batchCollectMs);
    state.batchEnqueueMs = shmEnqueueMs;
    state.batchCompleteMs = 0.0;
    state.claimTime = claimTime;
    state.firstResultTime = {};
  } else {
    state.batchEnqueueMs += shmEnqueueMs;
  }
}

void EmbeddingService::onTaskCompleted(
    size_t workerIdx, const IpcResult& result,
    std::chrono::steady_clock::time_point gotResultTime,
    std::chrono::steady_clock::time_point doneTime) {
  if (workerIdx >= workerBatchStates_.size()) return;
  auto& state = workerBatchStates_[workerIdx];
  if (state.firstResultTime.time_since_epoch().count() == 0) {
    state.firstResultTime = gotResultTime;
  }
  state.batchCompleteMs +=
      std::chrono::duration<double, std::milli>(doneTime - gotResultTime)
          .count();
  if (result.isBatchEnd() && state.batchSize > 0) {
    const double shmRoundTripMs =
        state.claimTime.time_since_epoch().count() > 0
            ? std::chrono::duration<double, std::milli>(state.firstResultTime -
                                                        state.claimTime)
                  .count()
            : 0.0;
    const bool summaryDue = dispatchSummary_.record(
        state.batchSize, state.queueWaitMs, state.batchCollectMs,
        state.batchEnqueueMs, shmRoundTripMs, state.batchCompleteMs);
    state.batchSize = 0;
    if (summaryDue) {
      dispatchSummary_.logSummary("periodic");
    }
  }
}

void EmbeddingService::onStopped() { dispatchSummary_.logSummary("final"); }

}  // namespace tt::services
