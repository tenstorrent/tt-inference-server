// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/mock_device_pipeline.hpp"

#include <algorithm>
#include <chrono>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/debug.hpp"

namespace sp_pipeline {

namespace {
constexpr uint64_t FALLBACK_TOKEN_ID = 12345;
}

MockDevicePipeline::MockDevicePipeline(MockDeviceConfig config)
    : config_(config) {
  pipeline_thread_ = std::thread([this] {
    tracy_config::TracySetThreadName("MockDevice::pipeline");
    pipeline_loop();
  });
}

MockDevicePipeline::~MockDevicePipeline() { exit(); }

void MockDevicePipeline::write(const std::string& task_id,
                               const std::vector<int64_t>& token_ids,
                               uint32_t max_tokens,
                               RequestPhase phase) {
  ZoneScopedN("MockDevice::write");
  auto req = std::make_unique<PipelineRequest>();
  req->task_id = task_id;
  req->token_ids = token_ids;
  req->max_tokens = max_tokens;
  req->is_decode = (phase == RequestPhase::DECODE);

  std::unique_lock lock(input_mutex_);
  input_not_full_.wait(lock, [this] {
    return input_queue_.size() < config_.write_queue_capacity ||
           stop_.load(std::memory_order_relaxed);
  });
  if (stop_.load(std::memory_order_relaxed)) return;
  input_queue_.push_back(std::move(req));
}

std::optional<llm_engine::TokenResult> MockDevicePipeline::read() {
  ZoneScopedN("MockDevice::read");
  std::unique_lock lock(output_mutex_);
  output_not_empty_.wait(lock, [this] {
    return !output_queue_.empty() ||
           stop_.load(std::memory_order_relaxed);
  });
  if (output_queue_.empty()) return std::nullopt;
  auto result = std::move(output_queue_.front());
  output_queue_.pop_front();
  return result;
}

void MockDevicePipeline::exit() {
  if (stop_.exchange(true)) return;
  input_not_full_.notify_all();
  output_not_empty_.notify_all();
  if (pipeline_thread_.joinable()) pipeline_thread_.join();
  LLM_ENGINE_LOG("mock_device_pipeline") << "exit" << std::endl;
}

// ---------------------------------------------------------------------------
// Pipeline internals — all called from pipeline_thread_ only,
// except the output queue push which is synchronised.
// ---------------------------------------------------------------------------

void MockDevicePipeline::drain_input() {
  std::lock_guard lock(input_mutex_);
  while (!input_queue_.empty()) {
    auto& req = input_queue_.front();
    if (req->is_decode) {
      decode_queue_.push_back(std::move(req));
    } else {
      prefill_queue_.push_back(std::move(req));
    }
    input_queue_.pop_front();
  }
  input_not_full_.notify_all();
}

void MockDevicePipeline::emit_token(RequestPtr& req) {
  uint64_t token_id =
      req->tokens_generated < req->token_ids.size()
          ? static_cast<uint64_t>(req->token_ids[req->tokens_generated])
          : FALLBACK_TOKEN_ID;
  ++req->tokens_generated;

  {
    std::lock_guard lock(output_mutex_);
    output_queue_.emplace_back(llm_engine::TaskID(req->task_id), token_id);
  }
  output_not_empty_.notify_one();
}

void MockDevicePipeline::handle_completion(RequestPtr req) {
  ZoneScopedN("MockDevice::handle_completion");

  emit_token(req);

  if (!req->is_decode) {
    req->is_decode = true;
  }

  if (req->tokens_generated < req->max_tokens) {
    decode_queue_.push_back(std::move(req));
  }
}

void MockDevicePipeline::insert_in_flight(InFlightRequest entry) {
  auto it = std::lower_bound(
      in_flight_pipeline_.begin(), in_flight_pipeline_.end(),
      entry.complete_at_tick,
      [](const InFlightRequest& e, size_t tick) {
        return e.complete_at_tick < tick;
      });
  in_flight_pipeline_.insert(it, std::move(entry));
}

MockDevicePipeline::RequestPtr MockDevicePipeline::schedule_next() {
  if (!decode_queue_.empty()) {
    auto req = std::move(decode_queue_.front());
    decode_queue_.pop_front();
    return req;
  }
  if (!prefill_queue_.empty()) {
    auto req = std::move(prefill_queue_.front());
    prefill_queue_.pop_front();
    return req;
  }
  return nullptr;
}

void MockDevicePipeline::try_schedule() {
  auto next = schedule_next();
  if (!next) return;

  active_req_ = std::move(next);
  if (active_req_->is_decode) {
    feed_remaining_ = 1;
  } else {
    size_t tokens_remaining =
        active_req_->token_ids.size() - active_req_->prefill_offset;
    size_t chunk_tokens =
        std::min(config_.prefill_chunk_size, tokens_remaining);
    active_req_->prefill_offset += static_cast<uint32_t>(chunk_tokens);
    feed_remaining_ = chunk_tokens;
  }
}

void MockDevicePipeline::pipeline_loop() {
  using clock = std::chrono::steady_clock;
  const auto tick_duration =
      std::chrono::microseconds(config_.stage_duration_us);

  while (!stop_.load(std::memory_order_relaxed)) {
    auto tick_start = clock::now();

    drain_input();

    while (!in_flight_pipeline_.empty() &&
           in_flight_pipeline_.front().complete_at_tick <= current_tick_) {
      handle_completion(std::move(in_flight_pipeline_.front().req));
      in_flight_pipeline_.pop_front();
    }

    if (active_req_ && !active_req_->is_decode && !decode_queue_.empty()) {
      active_req_->prefill_offset -= feed_remaining_;
      prefill_queue_.push_front(std::move(active_req_));
      active_req_ = nullptr;
      feed_remaining_ = 0;
    }

    if (!active_req_) {
      try_schedule();
    }

    if (active_req_) {
      --feed_remaining_;
      if (feed_remaining_ == 0) {
        bool is_intermediate_prefill =
            !active_req_->is_decode &&
            active_req_->prefill_offset < active_req_->token_ids.size();
        if (is_intermediate_prefill) {
          prefill_queue_.push_back(std::move(active_req_));
        } else {
          insert_in_flight(
              {current_tick_ + config_.num_stages, std::move(active_req_)});
        }
      }
    }

    ++current_tick_;

    while ((clock::now() - tick_start) < tick_duration) {
    }
  }
}

}  // namespace sp_pipeline
