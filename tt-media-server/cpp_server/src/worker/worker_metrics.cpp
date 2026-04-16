// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/worker_metrics.hpp"

#include <prometheus/gauge.h>
#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

#include <chrono>
#include <memory>
#include <sstream>

namespace tt::worker {

namespace {
struct PrometheusState {
  std::shared_ptr<prometheus::Registry> registry;
  prometheus::Gauge* stepHeartbeatAge{nullptr};
  prometheus::Gauge* outputHeartbeatAge{nullptr};
  prometheus::Gauge* activeRequestsGauge{nullptr};
  prometheus::Gauge* workerAlive{nullptr};
};

PrometheusState& promState() {
  static PrometheusState state;
  return state;
}
}  // namespace

WorkerMetrics& WorkerMetrics::instance() {
  static WorkerMetrics inst;
  return inst;
}

void WorkerMetrics::initialize(int workerId) {
  this->workerId = workerId;

  auto now = nowMs();
  stepEpochMs.store(now, std::memory_order_relaxed);
  lastOutputEpochMs.store(now, std::memory_order_relaxed);

  auto& ps = promState();
  ps.registry = std::make_shared<prometheus::Registry>();
  const std::string idStr = std::to_string(workerId);

  ps.stepHeartbeatAge =
      &prometheus::BuildGauge()
           .Name("tt_worker_heartbeat_age_seconds")
           .Help("Seconds since the worker last called step()")
           .Register(*ps.registry)
           .Add({{"worker_id", idStr}});

  ps.outputHeartbeatAge =
      &prometheus::BuildGauge()
           .Name("tt_worker_last_output_age_seconds")
           .Help("Seconds since the worker last produced a token")
           .Register(*ps.registry)
           .Add({{"worker_id", idStr}});

  ps.activeRequestsGauge =
      &prometheus::BuildGauge()
           .Name("tt_worker_active_requests")
           .Help("Number of requests currently in the worker pipeline")
           .Register(*ps.registry)
           .Add({{"worker_id", idStr}});

  ps.workerAlive = &prometheus::BuildGauge()
                        .Name("tt_worker_alive")
                        .Help("1 while the worker process is running")
                        .Register(*ps.registry)
                        .Add({{"worker_id", idStr}});
  ps.workerAlive->Set(1);

  initialized = true;
}

uint64_t WorkerMetrics::nowMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void WorkerMetrics::updateStepHeartbeat() {
  stepEpochMs.store(nowMs(), std::memory_order_relaxed);
}

void WorkerMetrics::updateOutputHeartbeat() {
  lastOutputEpochMs.store(nowMs(), std::memory_order_relaxed);
}

void WorkerMetrics::incrementActiveRequests() {
  uint32_t prev = activeRequestsCount.fetch_add(1, std::memory_order_relaxed);
  if (prev == 0) {
    lastOutputEpochMs.store(nowMs(), std::memory_order_relaxed);
  }
}

void WorkerMetrics::decrementActiveRequests() {
  activeRequestsCount.fetch_sub(1, std::memory_order_relaxed);
}

double WorkerMetrics::stepAgeSec() const {
  auto now = nowMs();
  auto last = stepEpochMs.load(std::memory_order_relaxed);
  return static_cast<double>(now - last) / 1000.0;
}

double WorkerMetrics::outputAgeSec() const {
  auto now = nowMs();
  auto last = lastOutputEpochMs.load(std::memory_order_relaxed);
  return static_cast<double>(now - last) / 1000.0;
}

uint32_t WorkerMetrics::activeRequests() const {
  return activeRequestsCount.load(std::memory_order_relaxed);
}

std::string WorkerMetrics::renderText() {
  if (!initialized) return "";

  auto& ps = promState();
  ps.stepHeartbeatAge->Set(stepAgeSec());
  ps.outputHeartbeatAge->Set(outputAgeSec());
  ps.activeRequestsGauge->Set(
      static_cast<double>(activeRequestsCount.load(std::memory_order_relaxed)));

  prometheus::TextSerializer serializer;
  std::ostringstream ss;
  serializer.Serialize(ss, ps.registry->Collect());
  return ss.str();
}

}  // namespace tt::worker
