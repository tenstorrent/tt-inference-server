// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "runners/llm_runner/scheduler.hpp"

namespace llm_engine {

/**
 * Decodes the current batch to completion before admitting new prefills.
 *
 * Prefill is only attempted when the running queue is empty (all previous
 * requests have finished). This mirrors the AscendScheduler approach where
 * running is filled up to capacity, decoded until requests finish and free
 * slots, and only then is the next batch admitted for prefill.
 */
class InterleavedScheduler : public Scheduler {
 public:
  using Scheduler::Scheduler;

 protected:
  bool should_prefill_first(bool has_waiting, int running_count,
                            int /*max_num_seqs*/) const override {
    if (!has_waiting) return false;
    return running_count == 0;
  }
};

}  // namespace llm_engine
