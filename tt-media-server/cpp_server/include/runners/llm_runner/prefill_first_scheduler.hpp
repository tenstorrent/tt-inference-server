// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "runners/llm_runner/scheduler.hpp"

namespace llm_engine {

/**
 * Always prefills when the waiting queue has requests (original behaviour).
 * Decode is only attempted when nothing can be prefilled.
 */
class PrefillFirstScheduler : public Scheduler {
 public:
  using Scheduler::Scheduler;

 protected:
  bool should_prefill_first(bool has_waiting, int /*running_count*/,
                            int /*max_num_seqs*/) const override {
    return has_waiting;
  }
};

}  // namespace llm_engine
