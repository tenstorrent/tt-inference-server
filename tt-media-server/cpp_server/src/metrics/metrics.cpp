// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "metrics/metrics.hpp"

#include <prometheus/text_serializer.h>

#include <sstream>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::metrics {

ServerMetrics& ServerMetrics::instance() {
  static ServerMetrics inst;
  return inst;
}

// Quantiles reported by the latency summaries (φ, ε) — exact values, 60 s
// window.
static const prometheus::Summary::Quantiles K_LATENCY_QUANTILES{
    {0.50, 0.01},
    {0.90, 0.005},
    {0.95, 0.005},
    {0.99, 0.001},
};

static const prometheus::Histogram::BucketBoundaries K_PROMPT_TOKEN_BUCKETS{
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4096, 8192};

static const prometheus::Histogram::BucketBoundaries K_GEN_TOKEN_BUCKETS{
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4096};

ServerMetrics::ServerMetrics() {
  model_name_ = tt::config::runnerType();
  registry_ = std::make_shared<prometheus::Registry>();

  const std::map<std::string, std::string> modelLabel = {
      {"model_name", model_name_}};

  // ----- counters --------------------------------------------------------
  prompt_tokens_total_ = &prometheus::BuildCounter()
                              .Name("tt_prompt_tokens_total")
                              .Help("Total number of prompt tokens processed")
                              .Register(*registry_)
                              .Add(modelLabel);

  generation_tokens_total_ =
      &prometheus::BuildCounter()
           .Name("tt_generation_tokens_total")
           .Help("Total number of generation tokens produced")
           .Register(*registry_)
           .Add(modelLabel);

  request_success_family_ =
      &prometheus::BuildCounter()
           .Name("tt_request_success_total")
           .Help("Number of completed requests labelled by finish reason")
           .Register(*registry_);

  // ----- gauges ----------------------------------------------------------
  queue_depth_ =
      &prometheus::BuildGauge()
           .Name("tt_num_requests_in_flight")
           .Help(
               "Number of requests in flight: from submission to final token "
               "delivery (includes queued, prefilling, and decoding)")
           .Register(*registry_)
           .Add({});

  max_queue_size_ = &prometheus::BuildGauge()
                         .Name("tt_max_queue_size")
                         .Help(
                             "Configured maximum number of requests that can "
                             "be queued (MAX_QUEUE_SIZE)")
                         .Register(*registry_)
                         .Add({});
  max_queue_size_->Set(static_cast<double>(tt::config::maxQueueSize()));

  decoding_requests_ =
      &prometheus::BuildGauge()
           .Name("tt_num_decoding_requests")
           .Help(
               "Number of requests actively generating tokens (excludes "
               "queued and prefilling requests)")
           .Register(*registry_)
           .Add({});

  // ----- latency summaries (exact quantiles, 60 s sliding window) ----------
  e2e_latency_seconds_ =
      &prometheus::BuildSummary()
           .Name("tt_e2e_request_latency_seconds")
           .Help("End-to-end request latency from submission to final token")
           .Register(*registry_)
           .Add(modelLabel, K_LATENCY_QUANTILES, std::chrono::seconds{60}, 5);

  ttft_seconds_ =
      &prometheus::BuildSummary()
           .Name("tt_time_to_first_token_seconds")
           .Help("Time from request submission to first generated token")
           .Register(*registry_)
           .Add(modelLabel, K_LATENCY_QUANTILES, std::chrono::seconds{60}, 5);

  inter_token_latency_seconds_ =
      &prometheus::BuildSummary()
           .Name("tt_inter_token_latency_seconds")
           .Help("Latency between consecutive generated tokens (decode step)")
           .Register(*registry_)
           .Add(modelLabel, K_LATENCY_QUANTILES, std::chrono::seconds{60}, 5);

  request_prompt_tokens_ =
      &prometheus::BuildHistogram()
           .Name("tt_request_prompt_tokens")
           .Help("Distribution of prompt token counts per request")
           .Register(*registry_)
           .Add(modelLabel, K_PROMPT_TOKEN_BUCKETS);

  request_generation_tokens_ =
      &prometheus::BuildHistogram()
           .Name("tt_request_generation_tokens")
           .Help("Distribution of generated token counts per request")
           .Register(*registry_)
           .Add(modelLabel, K_GEN_TOKEN_BUCKETS);

  // ----- start background metrics thread ----------------------------------
  running_ = true;
  metrics_thread_ = std::thread(&ServerMetrics::metricsLoop, this);
}

ServerMetrics::~ServerMetrics() {
  running_ = false;
  event_queue_cv_.notify_all();
  if (metrics_thread_.joinable()) metrics_thread_.join();
}

// -----------------------------------------------------------------------------
// Public API — capture timestamp immediately, push event, return.
// Zero prometheus work on the calling thread.
// -----------------------------------------------------------------------------

void ServerMetrics::onRequestSubmitted(uint32_t taskId, int promptTokens) {
  if (!tryPushEvent(EventRequestSubmitted{
          taskId, std::chrono::steady_clock::now(), promptTokens})) {
    TT_LOG_WARN("[ServerMetrics] event queue full, dropping RequestSubmitted");
  }
}

void ServerMetrics::onToken(uint32_t taskId) {
  if (!tryPushEvent(EventToken{taskId, std::chrono::steady_clock::now()})) {
    TT_LOG_WARN("[ServerMetrics] event queue full, dropping Token");
  }
}

void ServerMetrics::onRequestCompleted(uint32_t taskId,
                                       const std::string& finishReason) {
  if (!tryPushEvent(EventRequestCompleted{
          taskId, std::chrono::steady_clock::now(), finishReason})) {
    TT_LOG_WARN("[ServerMetrics] event queue full, dropping RequestCompleted");
  }
}

void ServerMetrics::setQueueDepth(double n) {
  // Called twice per request (not per token) — write directly.
  queue_depth_->Set(n);
}

// -----------------------------------------------------------------------------
// Queue helpers
// -----------------------------------------------------------------------------

bool ServerMetrics::tryPushEvent(MetricsEvent event) {
  std::lock_guard<std::mutex> lock(event_queue_mutex_);
  if (event_queue_.size() >= kMaxEventQueueSize) {
    return false;
  }
  const bool wasEmpty = event_queue_.empty();
  event_queue_.push(std::move(event));
  // Only wake the consumer when the queue transitions empty → non-empty.
  // Notifying on every push causes a futex syscall per token (~120k/s on the
  // mock runner), which is the dominant source of overhead.
  if (wasEmpty) event_queue_cv_.notify_one();
  return true;
}

// -----------------------------------------------------------------------------
// Background thread — sole consumer of the queue and contexts_ map.
// No locking needed for contexts_ or prometheus objects here.
// -----------------------------------------------------------------------------

void ServerMetrics::metricsLoop() {
  while (true) {
    std::queue<MetricsEvent> batch;
    {
      std::unique_lock<std::mutex> lock(event_queue_mutex_);
      event_queue_cv_.wait(
          lock, [this] { return !event_queue_.empty() || !running_.load(); });
      if (event_queue_.empty()) break;  // running_ == false and queue drained
      // Swap the entire queue out under the lock, then release immediately.
      // Producers can keep pushing while we process the batch.
      std::swap(batch, event_queue_);
    }
    while (!batch.empty()) {
      processEvent(batch.front());
      batch.pop();
    }
  }
}

void ServerMetrics::processEvent(const MetricsEvent& event) {
  std::visit(
      [this](const auto& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, EventRequestSubmitted>)
          handleRequestSubmitted(e);
        else if constexpr (std::is_same_v<T, EventToken>)
          handleToken(e);
        else if constexpr (std::is_same_v<T, EventRequestCompleted>)
          handleRequestCompleted(e);
      },
      event);
}

void ServerMetrics::handleRequestSubmitted(const EventRequestSubmitted& e) {
  contexts_.emplace(e.task_id, RequestContext{.start_time = e.time,
                                              .prev_token_time = {},
                                              .prompt_tokens = e.prompt_tokens,
                                              .generation_tokens = 0});
}

void ServerMetrics::handleToken(const EventToken& e) {
  auto it = contexts_.find(e.task_id);
  if (it == contexts_.end()) return;
  auto& ctx = it->second;

  if (ctx.generation_tokens == 0) {
    double ttft =
        std::chrono::duration<double>(e.time - ctx.start_time).count();
    ttft_seconds_->Observe(ttft);
    decoding_requests_->Increment();
  } else {
    double itl =
        std::chrono::duration<double>(e.time - ctx.prev_token_time).count();
    inter_token_latency_seconds_->Observe(itl);
  }
  ctx.prev_token_time = e.time;
  ctx.generation_tokens++;
}

void ServerMetrics::handleRequestCompleted(const EventRequestCompleted& e) {
  auto it = contexts_.find(e.task_id);
  if (it == contexts_.end()) return;
  const RequestContext ctx = it->second;
  contexts_.erase(it);

  e2e_latency_seconds_->Observe(
      std::chrono::duration<double>(e.time - ctx.start_time).count());

  if (ctx.generation_tokens > 0) decoding_requests_->Decrement();

  if (ctx.prompt_tokens > 0) {
    prompt_tokens_total_->Increment(ctx.prompt_tokens);
    request_prompt_tokens_->Observe(static_cast<double>(ctx.prompt_tokens));
  }
  if (ctx.generation_tokens > 0)
    generation_tokens_total_->Increment(ctx.generation_tokens);
  request_generation_tokens_->Observe(
      static_cast<double>(ctx.generation_tokens));

  request_success_family_
      ->Add({{"model_name", model_name_}, {"finished_reason", e.finish_reason}})
      .Increment();
}

// -----------------------------------------------------------------------------
// Scrape endpoint — prometheus internal locks make this safe to call from
// any thread while the metrics thread calls Observe()/Increment().
// -----------------------------------------------------------------------------

std::string ServerMetrics::renderText() const {
  prometheus::TextSerializer serializer;
  std::ostringstream ss;
  serializer.Serialize(ss, registry_->Collect());
  return ss.str();
}

}  // namespace tt::metrics
