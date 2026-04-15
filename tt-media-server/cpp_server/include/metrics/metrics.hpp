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
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>

namespace tt::metrics {

/**
 * Singleton Prometheus metrics registry for the TT inference server.
 *
 * All metric observations are handled by a dedicated background thread.
 * Callers (LLM service threads) only push lightweight event structs onto a
 * bounded queue — one atomic push per token, zero prometheus work on the
 * critical path.  If the queue is full, the event is dropped and a warning
 * is logged.
 *
 * setQueueDepth() is the exception: it writes a Gauge directly (called
 * twice per request, not per token — negligible overhead).
 *
 * Expose metrics via GET /metrics → renderText().
 */
class ServerMetrics {
 public:
  static ServerMetrics& instance();

  ServerMetrics(const ServerMetrics&) = delete;
  ServerMetrics& operator=(const ServerMetrics&) = delete;
  ~ServerMetrics();

  /**
   * Called when a streaming request is submitted to the service.
   * Starts the per-request timer and stores the prompt token count.
   */
  void onRequestSubmitted(uint32_t task_id, int prompt_tokens);

  /** Called for every generated token. Captures a timestamp on the caller's
   *  thread and pushes a lightweight event; the background thread derives
   *  TTFT, ITL, and token counts from the event stream. */
  void onToken(uint32_t task_id);

  /**
   * Called when a request produces its final token.
   * Records e2e latency, generation token count, and success counter.
   * Cleans up per-request state.
   */
  void onRequestCompleted(uint32_t task_id, const std::string& finish_reason);

  /**
   * Update the in-flight request gauge (call after pending_tasks_ changes).
   * Tracks all requests from submission through final token delivery.
   * Called directly (not via queue) — twice per request, not per token.
   */
  void setQueueDepth(double n);

  /** Render the full registry in Prometheus text exposition format. */
  std::string renderText() const;

 private:
  ServerMetrics();

  // -------------------------------------------------------------------------
  // Event types pushed onto the queue by LLM service threads.
  // Timestamps are captured at call time so measurements are accurate
  // even though processing happens asynchronously.
  // -------------------------------------------------------------------------
  struct EventRequestSubmitted {
    uint32_t task_id;
    std::chrono::steady_clock::time_point time;
    int prompt_tokens;
  };
  struct EventToken {
    uint32_t task_id;
    std::chrono::steady_clock::time_point time;
  };
  struct EventRequestCompleted {
    uint32_t task_id;
    std::chrono::steady_clock::time_point time;
    std::string finish_reason;
  };

  using MetricsEvent = std::variant<EventRequestSubmitted, EventToken,
                                    EventRequestCompleted>;

  // -------------------------------------------------------------------------
  // Background thread
  // -------------------------------------------------------------------------
  static constexpr size_t kMaxEventQueueSize = 65536;

  bool tryPushEvent(MetricsEvent event);
  void metricsLoop();
  void processEvent(const MetricsEvent& event);
  void handleRequestSubmitted(const EventRequestSubmitted& e);
  void handleToken(const EventToken& e);
  void handleRequestCompleted(const EventRequestCompleted& e);

  std::queue<MetricsEvent> event_queue_;
  std::mutex event_queue_mutex_;
  std::condition_variable event_queue_cv_;
  std::atomic<bool> running_{false};
  std::thread metrics_thread_;

  // -------------------------------------------------------------------------
  // Per-request timing state — only accessed from metrics_thread_.
  // No mutex needed: single consumer.
  // -------------------------------------------------------------------------
  struct RequestContext {
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point prev_token_time;
    int prompt_tokens = 0;
    int generation_tokens = 0;
  };
  std::unordered_map<uint32_t, RequestContext> contexts_;

  // -------------------------------------------------------------------------
  // Prometheus objects — only written from metrics_thread_,
  // read by renderText() (prometheus internal locks handle that safely).
  // -------------------------------------------------------------------------
  std::string model_name_;
  std::shared_ptr<prometheus::Registry> registry_;

  // --- counters ---
  prometheus::Counter* prompt_tokens_total_{nullptr};
  prometheus::Counter* generation_tokens_total_{nullptr};
  prometheus::Family<prometheus::Counter>* request_success_family_{nullptr};

  // --- gauges ---
  prometheus::Gauge* queue_depth_{nullptr};
  prometheus::Gauge* max_queue_size_{nullptr};
  prometheus::Gauge* decoding_requests_{nullptr};

  // --- latency summaries (exact quantiles via CKMS, 60 s sliding window) ---
  prometheus::Summary* e2e_latency_seconds_{nullptr};
  prometheus::Summary* ttft_seconds_{nullptr};
  prometheus::Summary* inter_token_latency_seconds_{nullptr};

  // --- token-count histograms ---
  prometheus::Histogram* request_prompt_tokens_{nullptr};
  prometheus::Histogram* request_generation_tokens_{nullptr};
};

}  // namespace tt::metrics
