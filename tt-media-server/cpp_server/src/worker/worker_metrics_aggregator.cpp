// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/worker_metrics_aggregator.hpp"

#include <sstream>

#include <vector>

#include "utils/logger.hpp"
#include "worker/worker_info.hpp"
#include "worker/worker_manager.hpp"

namespace tt::worker {

WorkerMetricsAggregator& WorkerMetricsAggregator::instance() {
  static WorkerMetricsAggregator inst;
  return inst;
}

void WorkerMetricsAggregator::initialize(const WorkerMetricsShmRegion* region,
                                         WorkerManager* mgr,
                                         size_t numWorkers) {
  region_ = region;
  mgr_ = mgr;
  numWorkers_ = numWorkers;
  registry_ = std::make_shared<prometheus::Registry>();
  initialized_ = true;
  TT_LOG_INFO(
      "[WorkerMetricsAggregator] Initialized for {} workers, region={}",
      numWorkers, static_cast<const void*>(region));
}

void WorkerMetricsAggregator::registerRenderer(
    MetricsLayout layout, std::unique_ptr<IWorkerMetricsRenderer> renderer) {
  by_layout_[layout] = std::move(renderer);
}

void WorkerMetricsAggregator::prebuildAll() {
  if (!initialized_ || registry_ == nullptr) return;
  for (size_t i = 0; i < numWorkers_; ++i) {
    for (auto& [layout, renderer] : by_layout_) {
      (void)layout;
      renderer->prebuildGauges(*registry_, static_cast<int>(i));
    }
  }
}

IWorkerMetricsRenderer* WorkerMetricsAggregator::rendererFor(
    MetricsLayout layout) {
  auto it = by_layout_.find(layout);
  if (it == by_layout_.end()) return nullptr;
  return it->second.get();
}

void WorkerMetricsAggregator::refresh() {
  if (!initialized_ || region_ == nullptr) return;
  std::lock_guard<std::mutex> lock(refresh_mutex_);

  std::vector<WorkerInfo> infos;
  if (mgr_ != nullptr) {
    infos = mgr_->getWorkerInfo();
  }

  for (size_t i = 0; i < numWorkers_; ++i) {
    const WorkerSlot& slot = region_->slots[i];
    uint32_t layoutTag = slot.metrics_layout.load(std::memory_order_acquire);
    auto layout = static_cast<MetricsLayout>(layoutTag);
    IWorkerMetricsRenderer* renderer = rendererFor(layout);
    if (renderer == nullptr) {
      // Forward-compat: a worker tagged with a layout this binary doesn't
      // know about (e.g. UNKNOWN before initialize, or a newer enum value)
      // is silently skipped instead of producing spurious gauges.
      continue;
    }
    bool is_alive = (i < infos.size()) && infos[i].is_alive;
    renderer->render(slot, static_cast<int>(i), is_alive);
  }
}

std::string WorkerMetricsAggregator::renderText() {
  if (!initialized_ || registry_ == nullptr) return "";
  prometheus::TextSerializer serializer;
  std::ostringstream ss;
  serializer.Serialize(ss, registry_->Collect());
  return ss.str();
}

}  // namespace tt::worker
