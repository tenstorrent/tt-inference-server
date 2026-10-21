// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/embedding_ipc_runner.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "config/settings.hpp"
#include "runtime/runners/runner_registry.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

namespace {

// Per-batch INFO lines are sampled so that a Galaxy BGE-large deployment
// (microsecond batches, >1M tok/s) is not slowed or flooded by logging, while a
// CI log still shows what batch sizes the worker actually formed: every batch
// during the first kInfoFirstBatches, then every kInfoEveryBatches-th one.
constexpr uint64_t kInfoFirstBatches = 20;
constexpr uint64_t kInfoEveryBatches = 50;

bool isSampledBatch(uint64_t batchNumber) {
  return batchNumber <= kInfoFirstBatches ||
         batchNumber % kInfoEveryBatches == 0;
}

}  // namespace

EmbeddingIpcRunner::EmbeddingIpcRunner(config::EmbeddingConfig cfg,
                                       int workerId)
    : config(std::move(cfg)), workerIndex(workerId) {
  taskQueue = std::make_unique<tt::ipc::embedding::EmbeddingTaskQueue>(
      tt::config::ttEmbeddingTaskQueueName());
  resultQueue = std::make_unique<tt::ipc::embedding::EmbeddingResultQueue>(
      std::string(tt::config::ttEmbeddingResultQueueName()) +
      std::to_string(workerIndex));
}

EmbeddingIpcRunner::~EmbeddingIpcRunner() { stop(); }

bool EmbeddingIpcRunner::warmup() {
  runner = tt::utils::RunnerRegistry::instance().createMedia<IEmbeddingRunner>(
      config::ModelService::EMBEDDING, config.runner_type,
      config::RunnerConfig{config});
  if (!runner) {
    throw std::runtime_error(
        "[EmbeddingIpcRunner] no embedding runner registered for runner_type=" +
        config::toString(config.runner_type));
  }
  TT_LOG_INFO("[EmbeddingIpcRunner] Worker {} warming embedding runner ({})",
              workerIndex, runner->runnerType());
  if (!runner->warmup()) {
    TT_LOG_ERROR("[EmbeddingIpcRunner] Worker {} warmup failed", workerIndex);
    throw std::runtime_error("[EmbeddingIpcRunner] warmup failed");
  }
  return true;
}

void EmbeddingIpcRunner::stop() {
  stopped.store(true, std::memory_order_release);
  if (runner) {
    runner->stop();
  }
}

void EmbeddingIpcRunner::run() {
  const size_t maxBatch = std::max<size_t>(1, config.max_batch_size);
  bool sawDone = false;
  uint64_t batchNumber = 0;

  TT_LOG_INFO(
      "[EmbeddingIpcRunner] Worker {} entering request loop "
      "(max_batch_size={})",
      workerIndex, maxBatch);

  while (!stopped.load(std::memory_order_acquire) && !sawDone) {
    while (!resultQueue->empty() && !stopped.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (stopped.load(std::memory_order_acquire)) {
      break;
    }

    tt::ipc::embedding::EmbeddingIpcTask first;
    taskQueue->receive(first);
    if (first.isDone()) {
      TT_LOG_INFO("[EmbeddingIpcRunner] Worker {} received shutdown task",
                  workerIndex);
      break;
    }

    const auto collectStart = std::chrono::steady_clock::now();
    std::vector<domain::EmbeddingRequest> batch;
    batch.reserve(maxBatch);
    batch.push_back(first.toDomain());

    const auto batchDeadline =
        collectStart + std::chrono::milliseconds(tt::config::batchTimeoutMs());
    while (batch.size() < maxBatch) {
      tt::ipc::embedding::EmbeddingIpcTask next;
      if (taskQueue->tryPop(next)) {
        if (next.isDone()) {
          sawDone = true;
          break;
        }
        batch.push_back(next.toDomain());
        continue;
      }
      if (maxBatch <= 1 || std::chrono::steady_clock::now() >= batchDeadline ||
          stopped.load(std::memory_order_acquire)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    const auto collectEnd = std::chrono::steady_clock::now();
    const float collectMs =
        std::chrono::duration<float, std::milli>(collectEnd - collectStart)
            .count();

    const auto batchSize = static_cast<uint32_t>(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      resultQueue->push(tt::ipc::embedding::EmbeddingIpcResult::claimed(
          batch[i].task_id, i == 0 ? batchSize : 0, i == 0 ? collectMs : 0.0f));
    }

    ++batchNumber;
    const bool sampled = isSampledBatch(batchNumber);
    const auto runStart = std::chrono::steady_clock::now();

    std::vector<domain::EmbeddingResponse> responses;
    try {
      responses = runner->run(batch);
      const float runMs = std::chrono::duration<float, std::milli>(
                              std::chrono::steady_clock::now() - runStart)
                              .count();
      if (sampled) {
        TT_LOG_INFO(
            "[EmbeddingIpcRunner] Worker {} batch #{}: {}/{} request(s), "
            "collected in {:.1f}ms, run {:.0f}ms",
            workerIndex, batchNumber, batchSize, maxBatch, collectMs, runMs);
      } else {
        TT_LOG_DEBUG(
            "[EmbeddingIpcRunner] Worker {} batch #{}: {}/{} request(s), "
            "collected in {:.1f}ms, run {:.0f}ms",
            workerIndex, batchNumber, batchSize, maxBatch, collectMs, runMs);
      }
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[EmbeddingIpcRunner] Worker {} batch failed: {}",
                   workerIndex, e.what());
      responses.clear();
      responses.reserve(batch.size());
      for (const auto& req : batch) {
        domain::EmbeddingResponse err(req.task_id);
        err.error = e.what();
        responses.push_back(std::move(err));
      }
    } catch (...) {
      TT_LOG_ERROR("[EmbeddingIpcRunner] Worker {} batch failed: unknown",
                   workerIndex);
      responses.clear();
      responses.reserve(batch.size());
      for (const auto& req : batch) {
        domain::EmbeddingResponse err(req.task_id);
        err.error = "unknown embedding runner error";
        responses.push_back(std::move(err));
      }
    }

    for (size_t i = 0; i < batch.size(); ++i) {
      const bool isBatchEnd = (i + 1 == batch.size());
      if (i < responses.size()) {
        resultQueue->push(tt::ipc::embedding::EmbeddingIpcResult::fromDomain(
            batch[i].task_id, responses[i], isBatchEnd));
      } else {
        domain::EmbeddingResponse missing(batch[i].task_id);
        missing.error = "No response from runner";
        resultQueue->push(tt::ipc::embedding::EmbeddingIpcResult::fromDomain(
            batch[i].task_id, missing, isBatchEnd));
      }
    }
  }

  if (runner) {
    runner->close();
  }
}

}  // namespace tt::runners
