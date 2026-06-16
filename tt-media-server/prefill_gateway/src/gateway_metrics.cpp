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
    std::lock_guard<std::mutex> lock(mutex);

    registry = std::make_shared<prometheus::Registry>();

    prefillCompletedFamily =
        &prometheus::BuildCounter()
             .Name("tt_prefill_completed_total")
             .Help(
                 "Prefill requests completed by the gateway, labelled by "
                 "prefill server and outcome.")
             .Register(*registry);
    routingDecisionsFamily =
        &prometheus::BuildCounter()
             .Name("tt_gateway_routing_decisions_total")
             .Help("Gateway routing decisions labelled by routing reason.")
             .Register(*registry);
    requestFailuresFamily =
        &prometheus::BuildCounter()
             .Name("tt_gateway_request_failures_total")
             .Help("Gateway request failures labelled by reason.")
             .Register(*registry);
    cancelsFamily = &prometheus::BuildCounter()
                           .Name("tt_gateway_cancels_total")
                           .Help("Gateway prefill cancel attempts.")
                           .Register(*registry);
    timeoutsFamily =
        &prometheus::BuildCounter()
             .Name("tt_gateway_prefill_timeouts_total")
             .Help("Gateway request timeouts labelled by prefill server.")
             .Register(*registry);
    prefillDownTasksTotal =
        &prometheus::BuildCounter()
             .Name("tt_gateway_prefill_down_tasks_failed_total")
             .Help("In-flight tasks failed because their prefill went down.")
             .Register(*registry)
             .Add({});
    cacheBlocksAddedTotal =
        &prometheus::BuildCounter()
             .Name("tt_gateway_cache_blocks_added_total")
             .Help("Cache block add notifications observed by the gateway.")
             .Register(*registry)
             .Add({});

    prefillInflightFamily =
        &prometheus::BuildGauge()
             .Name("tt_prefill_inflight")
             .Help("In-flight gateway requests per prefill server.")
             .Register(*registry);
    prefillHealthyFamily =
        &prometheus::BuildGauge()
             .Name("tt_prefill_healthy")
             .Help("Whether the gateway currently considers a prefill healthy.")
             .Register(*registry);
    prefillAcceptingFamily =
        &prometheus::BuildGauge()
             .Name("tt_prefill_accepting_tasks")
             .Help("Whether the gateway is routing new tasks to a prefill.")
             .Register(*registry);
    heartbeatAgeFamily =
        &prometheus::BuildGauge()
             .Name("tt_prefill_last_heartbeat_age_seconds")
             .Help(
                 "Seconds since the gateway last observed a prefill heartbeat.")
             .Register(*registry);
    cacheBlocksFamily =
        &prometheus::BuildGauge()
             .Name("tt_prefill_cache_blocks")
             .Help("Cache blocks known by the gateway per prefill server.")
             .Register(*registry);
    routingTableSize =
        &prometheus::BuildGauge()
             .Name("tt_gateway_routing_table_size")
             .Help(
                 "Number of cache block routing entries known by the gateway.")
             .Register(*registry)
             .Add({});
    decodeConnected =
        &prometheus::BuildGauge()
             .Name("tt_gateway_decode_connected")
             .Help("Whether the decode peer is currently connected.")
             .Register(*registry)
             .Add({});

    prefillLatencyFamily =
        &prometheus::BuildHistogram()
             .Name("tt_prefill_latency_seconds")
             .Help("Prefill request latency measured by the gateway.")
             .Register(*registry);
    prefixMatchDepth =
        &prometheus::BuildHistogram()
             .Name("tt_gateway_prefix_match_depth")
             .Help(
                 "Number of registration hashes carried by prefix-match "
                 "routing requests.")
             .Register(*registry)
             .Add({}, PREFIX_DEPTH_BUCKETS);

    completedByLabel.clear();
    routingByReason.clear();
    failuresByReason.clear();
    cancelsByResult.clear();
    timeoutsByPrefill.clear();
    inflightByPrefill.clear();
    healthyByPrefill.clear();
    acceptingByPrefill.clear();
    heartbeatByPrefill.clear();
    cacheBlocksByPrefill.clear();
    latencyByLabel.clear();
  }

  void recordRoutingDecision(std::string_view reason) {
    std::lock_guard<std::mutex> lock(mutex);
    counterFor(*routingDecisionsFamily, routingByReason,
               {{"reason", std::string(reason)}}, reason)
        .Increment();
  }

  void observePrefixMatchDepth(size_t depth) {
    std::lock_guard<std::mutex> lock(mutex);
    prefixMatchDepth->Observe(static_cast<double>(depth));
  }

  void setRoutingTableSize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    routingTableSize->Set(static_cast<double>(size));
  }

  void recordRequestCompleted(std::string_view serverId,
                              std::string_view outcome,
                              std::chrono::steady_clock::duration latency) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = labelKey(serverId, outcome);
    const std::map<std::string, std::string> labels{
        {"server_id", std::string(serverId)},
        {"outcome", std::string(outcome)}};
    counterFor(*prefillCompletedFamily, completedByLabel, labels, key)
        .Increment();
    histogramFor(*prefillLatencyFamily, latencyByLabel, labels, key,
                 PREFILL_LATENCY_BUCKETS)
        .Observe(std::chrono::duration<double>(latency).count());
  }

  void recordRequestFailed(std::string_view reason) {
    std::lock_guard<std::mutex> lock(mutex);
    counterFor(*requestFailuresFamily, failuresByReason,
               {{"reason", std::string(reason)}}, reason)
        .Increment();
  }

  void recordCancel(bool sent) {
    const std::string result = sent ? "sent" : "failed";
    std::lock_guard<std::mutex> lock(mutex);
    counterFor(*cancelsFamily, cancelsByResult, {{"result", result}},
               result)
        .Increment();
  }

  void recordTimeout(std::string_view serverId) {
    std::lock_guard<std::mutex> lock(mutex);
    counterFor(*timeoutsFamily, timeoutsByPrefill,
               {{"server_id", std::string(serverId)}}, serverId)
        .Increment();
  }

  void recordPrefillDownTasks(size_t count) {
    std::lock_guard<std::mutex> lock(mutex);
    prefillDownTasksTotal->Increment(static_cast<double>(count));
  }

  void recordCacheBlocksAdded(size_t count) {
    std::lock_guard<std::mutex> lock(mutex);
    cacheBlocksAddedTotal->Increment(static_cast<double>(count));
  }

  void setDecodeConnected(bool connected) {
    std::lock_guard<std::mutex> lock(mutex);
    decodeConnected->Set(boolToGauge(connected));
  }

  void setPrefillSnapshots(
      std::span<const GatewayPrefillMetricSnapshot> snapshots) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& snapshot : snapshots) {
      const std::map<std::string, std::string> labels{
          {"server_id", snapshot.serverId}};
      gaugeFor(*prefillInflightFamily, inflightByPrefill, labels,
               snapshot.serverId)
          .Set(static_cast<double>(snapshot.inFlight));
      gaugeFor(*prefillHealthyFamily, healthyByPrefill, labels,
               snapshot.serverId)
          .Set(boolToGauge(snapshot.healthy));
      gaugeFor(*prefillAcceptingFamily, acceptingByPrefill, labels,
               snapshot.serverId)
          .Set(boolToGauge(snapshot.acceptingTasks));
      gaugeFor(*heartbeatAgeFamily, heartbeatByPrefill, labels,
               snapshot.serverId)
          .Set(snapshot.heartbeat_age_seconds);
      gaugeFor(*cacheBlocksFamily, cacheBlocksByPrefill, labels,
               snapshot.serverId)
          .Set(static_cast<double>(snapshot.cachedBlocks));
    }
  }

  std::string renderText() const {
    std::lock_guard<std::mutex> lock(mutex);
    prometheus::TextSerializer serializer;
    std::ostringstream ss;
    serializer.Serialize(ss, registry->Collect());
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

  mutable std::mutex mutex;
  std::shared_ptr<prometheus::Registry> registry;

  prometheus::Family<prometheus::Counter>* prefillCompletedFamily{nullptr};
  prometheus::Family<prometheus::Counter>* routingDecisionsFamily{nullptr};
  prometheus::Family<prometheus::Counter>* requestFailuresFamily{nullptr};
  prometheus::Family<prometheus::Counter>* cancelsFamily{nullptr};
  prometheus::Family<prometheus::Counter>* timeoutsFamily{nullptr};
  prometheus::Counter* prefillDownTasksTotal{nullptr};
  prometheus::Counter* cacheBlocksAddedTotal{nullptr};

  prometheus::Family<prometheus::Gauge>* prefillInflightFamily{nullptr};
  prometheus::Family<prometheus::Gauge>* prefillHealthyFamily{nullptr};
  prometheus::Family<prometheus::Gauge>* prefillAcceptingFamily{nullptr};
  prometheus::Family<prometheus::Gauge>* heartbeatAgeFamily{nullptr};
  prometheus::Family<prometheus::Gauge>* cacheBlocksFamily{nullptr};
  prometheus::Gauge* routingTableSize{nullptr};
  prometheus::Gauge* decodeConnected{nullptr};

  prometheus::Family<prometheus::Histogram>* prefillLatencyFamily{nullptr};
  prometheus::Histogram* prefixMatchDepth{nullptr};

  std::unordered_map<std::string, prometheus::Counter*> completedByLabel;
  std::unordered_map<std::string, prometheus::Counter*> routingByReason;
  std::unordered_map<std::string, prometheus::Counter*> failuresByReason;
  std::unordered_map<std::string, prometheus::Counter*> cancelsByResult;
  std::unordered_map<std::string, prometheus::Counter*> timeoutsByPrefill;

  std::unordered_map<std::string, prometheus::Gauge*> inflightByPrefill;
  std::unordered_map<std::string, prometheus::Gauge*> healthyByPrefill;
  std::unordered_map<std::string, prometheus::Gauge*> acceptingByPrefill;
  std::unordered_map<std::string, prometheus::Gauge*> heartbeatByPrefill;
  std::unordered_map<std::string, prometheus::Gauge*> cacheBlocksByPrefill;
  std::unordered_map<std::string, prometheus::Histogram*> latencyByLabel;
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

void GatewayMetrics::setDecodeConnected(bool connected) {
  impl->setDecodeConnected(connected);
}

void GatewayMetrics::setPrefillSnapshots(
    std::span<const GatewayPrefillMetricSnapshot> snapshots) {
  impl->setPrefillSnapshots(snapshots);
}

std::string GatewayMetrics::renderText() const { return impl->renderText(); }

}  // namespace tt::gateway
