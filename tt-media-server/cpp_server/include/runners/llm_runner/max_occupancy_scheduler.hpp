// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "runners/llm_runner/scheduler.hpp"

namespace llm_engine {

/**
 * Keeps the device at full occupancy (max_num_seqs) whenever possible.
 *
 * When running sequences finish and free slots, this scheduler immediately
 * prefills enough new sequences to refill to max_num_seqs, then resumes
 * decode at full capacity. Inspired by vLLM's continuous batching, adapted
 * for pure prefill/decode batches.
 *
 * Trade-off: running sequences lose 1 decode step during each prefill
 * batch, but all subsequent decode steps run at full width.
 */
class MaxOccupancyScheduler : public Scheduler {
 public:
  using Scheduler::Scheduler;

 protected:
  bool should_prefill_first(bool has_waiting, int running_count,
                            int max_num_seqs) const override {
    if (!has_waiting) return false;
    return running_count < max_num_seqs;
  }

  int max_prefill_seqs(int running_count,
                       int max_num_seqs) const override {
    return max_num_seqs - running_count;
  }
};

}  // namespace llm_engine
