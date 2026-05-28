// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <prometheus/counter.h>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tt::gateway {

struct GatewayPrefillMetricSnapshot {
  std::string server_id;
  bool healthy = false;
  bool accepting_tasks = false;
  uint32_t in_flight = 0;
  size_t cached_blocks = 0;
  double heartbeat_age_seconds = 0.0;
};

class GatewayMetrics {
 public:
  static GatewayMetrics& instance();

  GatewayMetrics(const GatewayMetrics&) = delete;
  GatewayMetrics& operator=(const GatewayMetrics&) = delete;

  void recordRoutingDecision(std::string_view reason);
  void observePrefixMatchDepth(size_t depth);
  void setRoutingTableSize(size_t size);

  void recordRequestCompleted(std::string_view serverId,
                              std::string_view outcome,
                              std::chrono::steady_clock::duration latency);
  void recordRequestFailed(std::string_view reason);
  void recordCancel(bool sent);
  void recordTimeout(std::string_view serverId);
  void recordPrefillDownTasks(size_t count);
  void recordCacheBlocksAdded(size_t count);
  void recordCacheBlocksEvicted(size_t count);

  void setDecodeConnected(bool connected);
  void setPrefillSnapshots(
      const std::vector<GatewayPrefillMetricSnapshot>& snapshots);

  std::string renderText() const;
  void resetForTests();

 private:
  GatewayMetrics();

  prometheus::Counter& counterFor(
      prometheus::Family<prometheus::Counter>& family,
      std::unordered_map<std::string, prometheus::Counter*>& cache,
      const std::map<std::string, std::string>& labels,
      std::string_view cacheKey);
  prometheus::Gauge& gaugeFor(
      prometheus::Family<prometheus::Gauge>& family,
      std::unordered_map<std::string, prometheus::Gauge*>& cache,
      const std::map<std::string, std::string>& labels,
      std::string_view cacheKey);
  prometheus::Histogram& histogramFor(
      prometheus::Family<prometheus::Histogram>& family,
      std::unordered_map<std::string, prometheus::Histogram*>& cache,
      const std::map<std::string, std::string>& labels,
      std::string_view cacheKey,
      const prometheus::Histogram::BucketBoundaries& buckets);

  mutable std::mutex mutex_;
  std::shared_ptr<prometheus::Registry> registry_;

  prometheus::Family<prometheus::Counter>* prefill_completed_family_{nullptr};
  prometheus::Family<prometheus::Counter>* routing_decisions_family_{nullptr};
  prometheus::Family<prometheus::Counter>* request_failures_family_{nullptr};
  prometheus::Family<prometheus::Counter>* cancels_family_{nullptr};
  prometheus::Family<prometheus::Counter>* timeouts_family_{nullptr};
  prometheus::Counter* prefill_down_tasks_total_{nullptr};
  prometheus::Counter* cache_blocks_added_total_{nullptr};
  prometheus::Counter* cache_blocks_evicted_total_{nullptr};

  prometheus::Family<prometheus::Gauge>* prefill_inflight_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* prefill_healthy_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* prefill_accepting_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* heartbeat_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* cache_blocks_family_{nullptr};
  prometheus::Gauge* routing_table_size_{nullptr};
  prometheus::Gauge* decode_connected_{nullptr};

  prometheus::Family<prometheus::Histogram>* prefill_latency_family_{nullptr};
  prometheus::Histogram* prefix_match_depth_{nullptr};

  std::unordered_map<std::string, prometheus::Counter*> completed_by_label_;
  std::unordered_map<std::string, prometheus::Counter*> routing_by_reason_;
  std::unordered_map<std::string, prometheus::Counter*> failures_by_reason_;
  std::unordered_map<std::string, prometheus::Counter*> cancels_by_result_;
  std::unordered_map<std::string, prometheus::Counter*> timeouts_by_prefill_;

  std::unordered_map<std::string, prometheus::Gauge*> inflight_by_prefill_;
  std::unordered_map<std::string, prometheus::Gauge*> healthy_by_prefill_;
  std::unordered_map<std::string, prometheus::Gauge*> accepting_by_prefill_;
  std::unordered_map<std::string, prometheus::Gauge*> heartbeat_by_prefill_;
  std::unordered_map<std::string, prometheus::Gauge*> cache_blocks_by_prefill_;
  std::unordered_map<std::string, prometheus::Histogram*> latency_by_label_;
};

}  // namespace tt::gateway
