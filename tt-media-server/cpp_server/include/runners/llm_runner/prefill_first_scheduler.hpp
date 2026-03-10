// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "runners/llm_runner/scheduler.hpp"

namespace llm_engine {

/**
 * Always prefills when the prefill_queue has requests (original behaviour).
 * Decode is only attempted when nothing can be prefilled.
 */
class PrefillFirstScheduler : public Scheduler {
 public:
  using Scheduler::Scheduler;

 protected:
  bool should_prefill_first(int /*decode_count*/,
                            int /*max_num_seqs*/) const override {
    return true;
  }
};

}  // namespace llm_engine
