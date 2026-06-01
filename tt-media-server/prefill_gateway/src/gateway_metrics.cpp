// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_metrics.hpp"

#include <prometheus/counter.h>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace tt::gateway {
namespace {

const prometheus::Histogram::BucketBoundaries PREFILL_LATENCY_BUCKETS{
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1,  0.25,
    0.5,   1.0,   2.5,  5.0,   10.0, 30.0, 60.0};

const prometheus::Histogram::BucketBoundaries PREFIX_DEPTH_BUCKETS{
    0, 1, 2, 4, 8, 16, 32, 64};

std::string labelKey(std::string_view first, std::string_view second = {}) {
  std::string key(first);
  key.push_back('|');
  key.append(second);
  return key;
}

double boolToGauge(bool value) { return value ? 1.0 : 0.0; }

}  // namespace

class GatewayMetrics::Impl {
 public:
  Impl() { reset(); }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    registry_ = std::make_shared<prometheus::Registry>();

    prefill_completed_family_ =
        &prometheus::BuildCounter()
             .Name("tt_prefill_completed_total")
             .Help(
                 "Prefill requests completed by the gateway, labelled by "
                 "prefill server and outcome.")
             .Register(*registry_);
    routing_decisions_family_ =
        &prometheus::BuildCounter()
             .Name("tt_gateway_routing_decisions_total")
             .Help("Gateway routing decisions labelled by routing reason.")
             .Register(*registry_);
    request_failures_family_ =
        &prometheus::BuildCounter()
             .Name("tt_gateway_request_failures_total")
             .Help("Gateway request failures labelled by reason.")
             .Register(*registry_);
    cancels_family_ = &prometheus::BuildCounter()
                           .Name("tt_gateway_cancels_total")
                           .Help("Gateway prefill cancel attempts.")
                           .Register(*registry_);
    timeouts_family_ =
        &prometheus::BuildCounter()
             .Name("tt_gateway_prefill_timeouts_total")
             .Help("Gateway request timeouts labelled by prefill server.")
             .Register(*registry_);
    prefill_down_tasks_total_ =
        &prometheus::BuildCounter()
             .Name("tt_gateway_prefill_down_tasks_failed_total")
             .Help("In-flight tasks failed because their prefill went down.")
             .Register(*registry_)
             .Add({});
    cache_blocks_added_total_ =
        &prometheus::BuildCounter()
             .Name("tt_gateway_cache_blocks_added_total")
             .Help("Cache block add notifications observed by the gateway.")
             .Register(*registry_)
             .Add({});
    cache_blocks_evicted_total_ =
        &prometheus::BuildCounter()
             .Name("tt_gateway_cache_blocks_evicted_total")
             .Help(
                 "Cache block eviction notifications observed by the gateway.")
             .Register(*registry_)
             .Add({});

    prefill_inflight_family_ =
        &prometheus::BuildGauge()
             .Name("tt_prefill_inflight")
             .Help("In-flight gateway requests per prefill server.")
             .Register(*registry_);
    prefill_healthy_family_ =
        &prometheus::BuildGauge()
             .Name("tt_prefill_healthy")
             .Help("Whether the gateway currently considers a prefill healthy.")
             .Register(*registry_);
    prefill_accepting_family_ =
        &prometheus::BuildGauge()
             .Name("tt_prefill_accepting_tasks")
             .Help("Whether the gateway is routing new tasks to a prefill.")
             .Register(*registry_);
    heartbeat_age_family_ =
        &prometheus::BuildGauge()
             .Name("tt_prefill_last_heartbeat_age_seconds")
             .Help(
                 "Seconds since the gateway last observed a prefill heartbeat.")
             .Register(*registry_);
    cache_blocks_family_ =
        &prometheus::BuildGauge()
             .Name("tt_prefill_cache_blocks")
             .Help("Cache blocks known by the gateway per prefill server.")
             .Register(*registry_);
    routing_table_size_ =
        &prometheus::BuildGauge()
             .Name("tt_gateway_routing_table_size")
             .Help("Number of affinity routing entries known by the gateway.")
             .Register(*registry_)
             .Add({});
    decode_connected_ =
        &prometheus::BuildGauge()
             .Name("tt_gateway_decode_connected")
             .Help("Whether the decode peer is currently connected.")
             .Register(*registry_)
             .Add({});

    prefill_latency_family_ =
        &prometheus::BuildHistogram()
             .Name("tt_prefill_latency_seconds")
             .Help("Prefill request latency measured by the gateway.")
             .Register(*registry_);
    prefix_match_depth_ =
        &prometheus::BuildHistogram()
             .Name("tt_gateway_prefix_match_depth")
             .Help(
                 "Number of registration hashes carried by prefix-match "
                 "routing requests.")
             .Register(*registry_)
             .Add({}, PREFIX_DEPTH_BUCKETS);

    completed_by_label_.clear();
    routing_by_reason_.clear();
    failures_by_reason_.clear();
    cancels_by_result_.clear();
    timeouts_by_prefill_.clear();
    inflight_by_prefill_.clear();
    healthy_by_prefill_.clear();
    accepting_by_prefill_.clear();
    heartbeat_by_prefill_.clear();
    cache_blocks_by_prefill_.clear();
    latency_by_label_.clear();
  }

  void recordRoutingDecision(std::string_view reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    counterFor(*routing_decisions_family_, routing_by_reason_,
               {{"reason", std::string(reason)}}, reason)
        .Increment();
  }

  void observePrefixMatchDepth(size_t depth) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefix_match_depth_->Observe(static_cast<double>(depth));
  }

  void setRoutingTableSize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    routing_table_size_->Set(static_cast<double>(size));
  }

  void recordRequestCompleted(std::string_view serverId,
                              std::string_view outcome,
                              std::chrono::steady_clock::duration latency) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto key = labelKey(serverId, outcome);
    const std::map<std::string, std::string> labels{
        {"server_id", std::string(serverId)},
        {"outcome", std::string(outcome)}};
    counterFor(*prefill_completed_family_, completed_by_label_, labels, key)
        .Increment();
    histogramFor(*prefill_latency_family_, latency_by_label_, labels, key,
                 PREFILL_LATENCY_BUCKETS)
        .Observe(std::chrono::duration<double>(latency).count());
  }

  void recordRequestFailed(std::string_view reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    counterFor(*request_failures_family_, failures_by_reason_,
               {{"reason", std::string(reason)}}, reason)
        .Increment();
  }

  void recordCancel(bool sent) {
    const std::string result = sent ? "sent" : "failed";
    std::lock_guard<std::mutex> lock(mutex_);
    counterFor(*cancels_family_, cancels_by_result_, {{"result", result}},
               result)
        .Increment();
  }

  void recordTimeout(std::string_view serverId) {
    std::lock_guard<std::mutex> lock(mutex_);
    counterFor(*timeouts_family_, timeouts_by_prefill_,
               {{"server_id", std::string(serverId)}}, serverId)
        .Increment();
  }

  void recordPrefillDownTasks(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefill_down_tasks_total_->Increment(static_cast<double>(count));
  }

  void recordCacheBlocksAdded(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_blocks_added_total_->Increment(static_cast<double>(count));
  }

  void recordCacheBlocksEvicted(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_blocks_evicted_total_->Increment(static_cast<double>(count));
  }

  void setDecodeConnected(bool connected) {
    std::lock_guard<std::mutex> lock(mutex_);
    decode_connected_->Set(boolToGauge(connected));
  }

  void setPrefillSnapshots(
      std::span<const GatewayPrefillMetricSnapshot> snapshots) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& snapshot : snapshots) {
      const std::map<std::string, std::string> labels{
          {"server_id", snapshot.server_id}};
      gaugeFor(*prefill_inflight_family_, inflight_by_prefill_, labels,
               snapshot.server_id)
          .Set(static_cast<double>(snapshot.in_flight));
      gaugeFor(*prefill_healthy_family_, healthy_by_prefill_, labels,
               snapshot.server_id)
          .Set(boolToGauge(snapshot.healthy));
      gaugeFor(*prefill_accepting_family_, accepting_by_prefill_, labels,
               snapshot.server_id)
          .Set(boolToGauge(snapshot.accepting_tasks));
      gaugeFor(*heartbeat_age_family_, heartbeat_by_prefill_, labels,
               snapshot.server_id)
          .Set(snapshot.heartbeat_age_seconds);
      gaugeFor(*cache_blocks_family_, cache_blocks_by_prefill_, labels,
               snapshot.server_id)
          .Set(static_cast<double>(snapshot.cached_blocks));
    }
  }

  std::string renderText() const {
    std::lock_guard<std::mutex> lock(mutex_);
    prometheus::TextSerializer serializer;
    std::ostringstream ss;
    serializer.Serialize(ss, registry_->Collect());
    return ss.str();
  }

  static prometheus::Counter& counterFor(
      prometheus::Family<prometheus::Counter>& family,
      std::unordered_map<std::string, prometheus::Counter*>& cache,
      const std::map<std::string, std::string>& labels,
      std::string_view cacheKey) {
    auto key = std::string(cacheKey);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto* counter = &family.Add(labels);
    cache.emplace(std::move(key), counter);
    return *counter;
  }

  static prometheus::Gauge& gaugeFor(
      prometheus::Family<prometheus::Gauge>& family,
      std::unordered_map<std::string, prometheus::Gauge*>& cache,
      const std::map<std::string, std::string>& labels,
      std::string_view cacheKey) {
    auto key = std::string(cacheKey);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto* gauge = &family.Add(labels);
    cache.emplace(std::move(key), gauge);
    return *gauge;
  }

  static prometheus::Histogram& histogramFor(
      prometheus::Family<prometheus::Histogram>& family,
      std::unordered_map<std::string, prometheus::Histogram*>& cache,
      const std::map<std::string, std::string>& labels,
      std::string_view cacheKey,
      const prometheus::Histogram::BucketBoundaries& buckets) {
    auto key = std::string(cacheKey);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto* histogram = &family.Add(labels, buckets);
    cache.emplace(std::move(key), histogram);
    return *histogram;
  }

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

GatewayMetrics& GatewayMetrics::instance() {
  static GatewayMetrics inst;
  return inst;
}

GatewayMetrics::GatewayMetrics() : impl(std::make_unique<Impl>()) {}

GatewayMetrics::~GatewayMetrics() = default;

void GatewayMetrics::resetForTests() { impl->reset(); }

void GatewayMetrics::recordRoutingDecision(std::string_view reason) {
  impl->recordRoutingDecision(reason);
}

void GatewayMetrics::observePrefixMatchDepth(size_t depth) {
  impl->observePrefixMatchDepth(depth);
}

void GatewayMetrics::setRoutingTableSize(size_t size) {
  impl->setRoutingTableSize(size);
}

void GatewayMetrics::recordRequestCompleted(
    std::string_view serverId, std::string_view outcome,
    std::chrono::steady_clock::duration latency) {
  impl->recordRequestCompleted(serverId, outcome, latency);
}

void GatewayMetrics::recordRequestFailed(std::string_view reason) {
  impl->recordRequestFailed(reason);
}

void GatewayMetrics::recordCancel(bool sent) { impl->recordCancel(sent); }

void GatewayMetrics::recordTimeout(std::string_view serverId) {
  impl->recordTimeout(serverId);
}

void GatewayMetrics::recordPrefillDownTasks(size_t count) {
  impl->recordPrefillDownTasks(count);
}

void GatewayMetrics::recordCacheBlocksAdded(size_t count) {
  impl->recordCacheBlocksAdded(count);
}

void GatewayMetrics::recordCacheBlocksEvicted(size_t count) {
  impl->recordCacheBlocksEvicted(count);
}

void GatewayMetrics::setDecodeConnected(bool connected) {
  impl->setDecodeConnected(connected);
}

void GatewayMetrics::setPrefillSnapshots(
    std::span<const GatewayPrefillMetricSnapshot> snapshots) {
  impl->setPrefillSnapshots(snapshots);
}

std::string GatewayMetrics::renderText() const { return impl->renderText(); }

}  // namespace tt::gateway
