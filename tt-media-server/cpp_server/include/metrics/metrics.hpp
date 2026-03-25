// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <prometheus/counter.h>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <prometheus/summary.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace tt::metrics {

/**
 * Singleton Prometheus metrics registry for the TT inference server.
 *
 * Tracks latency histograms (e2e, TTFT, inter-token), token throughput
 * counters, and system state gauges.  Thread-safe: all public methods
 * may be called concurrently from IO threads and consumer threads.
 *
 * Expose via GET /metrics → renderText().
 */
class ServerMetrics {
 public:
  static ServerMetrics& instance();

  ServerMetrics(const ServerMetrics&) = delete;
  ServerMetrics& operator=(const ServerMetrics&) = delete;

  /**
   * Called when a streaming request is submitted to the service.
   * Starts the per-request timer and stores the prompt token count.
   */
  void onRequestSubmitted(const std::string& task_id, int prompt_tokens);

  /**
   * Called for every generated token (including filtered reasoning tokens).
   * Records inter-token latency and TTFT on the first call per request.
   */
  void onToken(const std::string& task_id);

  /**
   * Called when a request produces its final token.
   * Records e2e latency, generation token count, and success counter.
   * Cleans up per-request state.
   */
  void onRequestCompleted(const std::string& task_id,
                          const std::string& finish_reason);

  /**
   * Update the in-flight request gauge (call after pending_tasks_ changes).
   * Tracks all requests from submission through final token delivery.
   */
  void setNumRequestsInFlight(double n);

  /** Render the full registry in Prometheus text exposition format. */
  std::string renderText() const;

 private:
  ServerMetrics();

  struct RequestContext {
    std::chrono::steady_clock::time_point start_time;
    std::optional<std::chrono::steady_clock::time_point> first_token_time;
    std::optional<std::chrono::steady_clock::time_point> prev_token_time;
    int prompt_tokens = 0;
    int generation_tokens = 0;
  };

  std::string model_name_;
  std::shared_ptr<prometheus::Registry> registry_;

  // --- counters ---
  prometheus::Counter* prompt_tokens_total_{nullptr};
  prometheus::Counter* generation_tokens_total_{nullptr};
  prometheus::Family<prometheus::Counter>* request_success_family_{nullptr};

  // --- gauges ---
  prometheus::Gauge* num_requests_in_flight_{nullptr};
  prometheus::Gauge* max_queue_size_{nullptr};

  // --- latency summaries (exact quantiles via CKMS, 60 s sliding window) ---
  prometheus::Summary* e2e_latency_seconds_{nullptr};
  prometheus::Summary* ttft_seconds_{nullptr};
  prometheus::Summary* inter_token_latency_seconds_{nullptr};

  // --- token-count histograms ---
  prometheus::Histogram* request_prompt_tokens_{nullptr};
  prometheus::Histogram* request_generation_tokens_{nullptr};

  // per-request timing / token-count state
  mutable std::mutex contexts_mutex_;
  std::unordered_map<std::string, RequestContext> contexts_;
};

}  // namespace tt::metrics
