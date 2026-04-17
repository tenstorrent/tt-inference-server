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

void WorkerMetricsAggregator::initialize(
    const WorkerMetricsShmRegion* region, WorkerManager* mgr,
    std::vector<MetricsLayout> layout_by_worker) {
  region_ = region;
  mgr_ = mgr;
  layout_by_worker_ = std::move(layout_by_worker);
  renderer_by_worker_.assign(layout_by_worker_.size(), nullptr);
  layout_tags_verified_ = false;
  registry_ = std::make_shared<prometheus::Registry>();
  initialized_ = true;
  TT_LOG_INFO(
      "[WorkerMetricsAggregator] Initialized for {} workers, region={}",
      layout_by_worker_.size(), static_cast<const void*>(region));
}

void WorkerMetricsAggregator::registerRenderer(
    MetricsLayout layout, std::unique_ptr<IWorkerMetricsRenderer> renderer) {
  by_layout_[layout] = std::move(renderer);
}

void WorkerMetricsAggregator::prebuildAll() {
  if (!initialized_ || registry_ == nullptr) return;
  for (size_t i = 0; i < layout_by_worker_.size(); ++i) {
    IWorkerMetricsRenderer* renderer = rendererFor(layout_by_worker_[i]);
    renderer_by_worker_[i] = renderer;
    if (renderer == nullptr) {
      TT_LOG_WARN(
          "[WorkerMetricsAggregator] No renderer registered for worker {} "
          "(layout={}); its metrics will not be exported",
          i, static_cast<uint32_t>(layout_by_worker_[i]));
      continue;
    }
    renderer->prebuildGauges(*registry_, static_cast<int>(i));
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

  // One-time sanity check that what each worker actually wrote into its
  // slot tag matches what main configured it for. This catches
  // config/runner-code drift (it can never disagree at runtime otherwise
  // because main and worker are the same binary).
  if (!layout_tags_verified_) {
    bool all_attached = true;
    for (size_t i = 0; i < layout_by_worker_.size(); ++i) {
      uint8_t tag = region_->slots[i].metrics_layout.load(
          std::memory_order_acquire);
      if (tag == static_cast<uint8_t>(MetricsLayout::UNKNOWN)) {
        all_attached = false;  // worker hasn't attached yet, retry next scrape
        continue;
      }
      if (tag != static_cast<uint8_t>(layout_by_worker_[i])) {
        TT_LOG_ERROR(
            "[WorkerMetricsAggregator] Worker {} layout tag mismatch: slot "
            "says {}, main configured {}",
            i, static_cast<uint32_t>(tag),
            static_cast<uint32_t>(layout_by_worker_[i]));
      }
    }
    if (all_attached) {
      layout_tags_verified_ = true;
    }
  }

  std::vector<WorkerInfo> infos;
  if (mgr_ != nullptr) {
    infos = mgr_->getWorkerInfo();
  }

  for (size_t i = 0; i < renderer_by_worker_.size(); ++i) {
    IWorkerMetricsRenderer* renderer = renderer_by_worker_[i];
    if (renderer == nullptr) continue;
    bool is_alive = (i < infos.size()) && infos[i].is_alive;
    renderer->render(region_->slots[i], static_cast<int>(i), is_alive);
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
