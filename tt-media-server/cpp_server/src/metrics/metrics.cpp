// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "metrics/metrics.hpp"

#include <prometheus/text_serializer.h>

#include <sstream>

#include "config/settings.hpp"

namespace tt::metrics {

ServerMetrics& ServerMetrics::instance() {
  static ServerMetrics inst;
  return inst;
}

// Quantiles reported by the latency summaries (φ, ε) — exact values, 60 s window.
static const prometheus::Summary::Quantiles kLatencyQuantiles{
    {0.50, 0.01},
    {0.90, 0.005},
    {0.95, 0.005},
    {0.99, 0.001},
};

static const prometheus::Histogram::BucketBoundaries kPromptTokenBuckets{
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4096, 8192};

static const prometheus::Histogram::BucketBoundaries kGenTokenBuckets{
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4096};

ServerMetrics::ServerMetrics() {
  model_name_ = tt::config::runnerType();
  registry_ = std::make_shared<prometheus::Registry>();

  const std::map<std::string, std::string> model_label = {
      {"model_name", model_name_}};

  // ----- counters --------------------------------------------------------
  prompt_tokens_total_ =
      &prometheus::BuildCounter()
           .Name("tt_prompt_tokens_total")
           .Help("Total number of prompt tokens processed")
           .Register(*registry_)
           .Add(model_label);

  generation_tokens_total_ =
      &prometheus::BuildCounter()
           .Name("tt_generation_tokens_total")
           .Help("Total number of generation tokens produced")
           .Register(*registry_)
           .Add(model_label);

  request_success_family_ =
      &prometheus::BuildCounter()
           .Name("tt_request_success_total")
           .Help("Number of completed requests labelled by finish reason")
           .Register(*registry_);

  // ----- gauges ----------------------------------------------------------
  num_requests_in_flight_ =
      &prometheus::BuildGauge()
           .Name("tt_num_requests_in_flight")
           .Help("Number of requests in flight: from submission to final token delivery (includes queued, prefilling, and decoding)")
           .Register(*registry_)
           .Add({});

  max_queue_size_ =
      &prometheus::BuildGauge()
           .Name("tt_max_queue_size")
           .Help("Configured maximum number of requests that can be queued (MAX_QUEUE_SIZE)")
           .Register(*registry_)
           .Add({});
  max_queue_size_->Set(static_cast<double>(tt::config::maxQueueSize()));

  // ----- latency summaries (exact quantiles, 60 s sliding window) ----------
  e2e_latency_seconds_ =
      &prometheus::BuildSummary()
           .Name("tt_e2e_request_latency_seconds")
           .Help("End-to-end request latency from submission to final token")
           .Register(*registry_)
           .Add(model_label, kLatencyQuantiles, std::chrono::seconds{60}, 5);

  ttft_seconds_ =
      &prometheus::BuildSummary()
           .Name("tt_time_to_first_token_seconds")
           .Help("Time from request submission to first generated token")
           .Register(*registry_)
           .Add(model_label, kLatencyQuantiles, std::chrono::seconds{60}, 5);

  inter_token_latency_seconds_ =
      &prometheus::BuildSummary()
           .Name("tt_inter_token_latency_seconds")
           .Help("Latency between consecutive generated tokens (decode step)")
           .Register(*registry_)
           .Add(model_label, kLatencyQuantiles, std::chrono::seconds{60}, 5);

  request_prompt_tokens_ =
      &prometheus::BuildHistogram()
           .Name("tt_request_prompt_tokens")
           .Help("Distribution of prompt token counts per request")
           .Register(*registry_)
           .Add(model_label, kPromptTokenBuckets);

  request_generation_tokens_ =
      &prometheus::BuildHistogram()
           .Name("tt_request_generation_tokens")
           .Help("Distribution of generated token counts per request")
           .Register(*registry_)
           .Add(model_label, kGenTokenBuckets);
}

void ServerMetrics::onRequestSubmitted(const std::string& task_id,
                                       int prompt_tokens) {
  std::lock_guard<std::mutex> lock(contexts_mutex_);
  contexts_.emplace(task_id, RequestContext{
                                 .start_time = std::chrono::steady_clock::now(),
                                 .prompt_tokens = prompt_tokens});
}

void ServerMetrics::onToken(const std::string& task_id) {
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(contexts_mutex_);
  auto it = contexts_.find(task_id);
  if (it == contexts_.end()) return;

  auto& ctx = it->second;
  ctx.generation_tokens++;
  generation_tokens_total_->Increment();

  if (!ctx.first_token_time.has_value()) {
    ctx.first_token_time = now;
    double ttft =
        std::chrono::duration<double>(now - ctx.start_time).count();
    ttft_seconds_->Observe(ttft);
  } else if (ctx.prev_token_time.has_value()) {
    double itl = std::chrono::duration<double>(
                     now - ctx.prev_token_time.value())
                     .count();
    inter_token_latency_seconds_->Observe(itl);
  }
  ctx.prev_token_time = now;
}

void ServerMetrics::onRequestCompleted(const std::string& task_id,
                                       const std::string& finish_reason) {
  auto now = std::chrono::steady_clock::now();
  RequestContext ctx;
  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto it = contexts_.find(task_id);
    if (it == contexts_.end()) return;
    ctx = it->second;
    contexts_.erase(it);
  }

  // E2E latency
  double e2e = std::chrono::duration<double>(now - ctx.start_time).count();
  e2e_latency_seconds_->Observe(e2e);

  // Prompt tokens
  if (ctx.prompt_tokens > 0) {
    prompt_tokens_total_->Increment(ctx.prompt_tokens);
    request_prompt_tokens_->Observe(static_cast<double>(ctx.prompt_tokens));
  }

  // Generation tokens
  request_generation_tokens_->Observe(
      static_cast<double>(ctx.generation_tokens));

  // Request success counter (label per finish_reason)
  request_success_family_
      ->Add({{"model_name", model_name_}, {"finished_reason", finish_reason}})
      .Increment();
}

void ServerMetrics::setNumRequestsInFlight(double n) {
  num_requests_in_flight_->Set(n);
}


std::string ServerMetrics::renderText() const {
  prometheus::TextSerializer serializer;
  std::ostringstream ss;
  serializer.Serialize(ss, registry_->Collect());
  return ss.str();
}

}  // namespace tt::metrics
