// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "runners/llm_runner/scheduler.hpp"

namespace llm_engine {

/**
 * Keeps the device at full occupancy (max_in_flight_count) whenever possible.
 *
 * When decode sequences finish and free slots, this scheduler immediately
 * prefills enough new sequences to refill to max_in_flight_count, then resumes
 * decode at full capacity. Inspired by vLLM's continuous batching, adapted
 * for pure prefill/decode batches.
 *
 * Trade-off: decode sequences lose 1 decode step during each prefill
 * batch, but all subsequent decode steps run at full width.
 *
 * Use for: High request rates, throughput-oriented applications.
 * Provides better average TTFT across all users under high load by
 * maximizing device utilization and keeping decode batches at full capacity.
 */
class MaxOccupancyScheduler : public Scheduler {
 public:
  using Scheduler::Scheduler;

 protected:
  bool shouldPrefillFirst(size_t decodeCount,
                          size_t maxInFlightCount) const override {
    return decodeCount < maxInFlightCount;
  }

  size_t maxPrefillSeqs(size_t decodeCount,
                        size_t maxInFlightCount) const override {
    return maxInFlightCount - decodeCount;
  }
};

}  // namespace llm_engine
