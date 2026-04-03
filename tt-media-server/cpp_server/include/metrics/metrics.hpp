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

  /**
   * Called for the first generated token of a request.
   * Records time-to-first-token (TTFT).
   *
   * Only the first token needs to be reported here; subsequent tokens are
   * handled via onITLSample() at a reduced rate.
   */
  void onToken(uint32_t task_id);

  /**
   * Called for a sampled inter-token latency observation.
   *
   * The caller (LLMService) is responsible for:
   *   - Computing the elapsed time between two consecutive tokens.
   *   - Calling this method only every kItlSampleStride tokens (defined in
   *     llm_service.cpp) to amortise the per-event cost.
   *
   * itl_seconds must be the actual elapsed time between two CONSECUTIVE tokens
   * (not between two sampled tokens), computed by the caller before invoking
   * this method.
   */
  void onITLSample(uint32_t task_id, double itl_seconds);

  /**
   * Called when a request produces its final token.
   * Records e2e latency, generation token count, and success counter.
   * Cleans up per-request state.
   *
   * generation_tokens is passed explicitly by the caller because token events
   * are sampled (only a fraction reach the metrics queue).
   */
  void onRequestCompleted(uint32_t task_id, const std::string& finish_reason,
                          int generation_tokens);

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
  // Fired once per request (first token only) — drives TTFT.
  struct EventFirstToken {
    uint32_t task_id;
    std::chrono::steady_clock::time_point time;
  };
  // Fired every kItlSampleStride tokens (see llm_service.cpp).
  // itl_seconds is the actual elapsed time between two CONSECUTIVE tokens,
  // pre-computed by the caller so this thread needs no per-task clock state.
  struct EventITLSample {
    uint32_t task_id;
    double itl_seconds;
  };
  struct EventRequestCompleted {
    uint32_t task_id;
    std::chrono::steady_clock::time_point time;
    std::string finish_reason;
    // Passed explicitly because token events are sampled, so the background
    // thread cannot count tokens from the queue alone.
    int generation_tokens;
  };

  using MetricsEvent = std::variant<EventRequestSubmitted, EventFirstToken,
                                    EventITLSample, EventRequestCompleted>;

  // -------------------------------------------------------------------------
  // Background thread
  // -------------------------------------------------------------------------
  static constexpr size_t kMaxEventQueueSize = 65536;

  bool tryPushEvent(MetricsEvent event);
  void metricsLoop();
  void processEvent(const MetricsEvent& event);
  void handleRequestSubmitted(const EventRequestSubmitted& e);
  void handleFirstToken(const EventFirstToken& e);
  void handleITLSample(const EventITLSample& e);
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
    std::optional<std::chrono::steady_clock::time_point> first_token_time;
    int prompt_tokens = 0;
    // generation_tokens is no longer tracked here: it arrives with
    // EventRequestCompleted, supplied by the caller (LLMService).
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

  // --- latency summaries (exact quantiles via CKMS, 60 s sliding window) ---
  prometheus::Summary* e2e_latency_seconds_{nullptr};
  prometheus::Summary* ttft_seconds_{nullptr};
  prometheus::Summary* inter_token_latency_seconds_{nullptr};

  // --- token-count histograms ---
  prometheus::Histogram* request_prompt_tokens_{nullptr};
  prometheus::Histogram* request_generation_tokens_{nullptr};
};

}  // namespace tt::metrics
