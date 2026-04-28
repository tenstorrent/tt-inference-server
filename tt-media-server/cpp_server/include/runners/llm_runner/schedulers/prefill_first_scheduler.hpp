// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runners/llm_runner/schedulers/scheduler.hpp"

namespace tt::runners::schedulers {

/**
 * Always prefills when the prefill_queue has requests (original behaviour).
 * Decode is only attempted when nothing can be prefilled.
 *
 * Use for: Low to moderate request rates, latency-sensitive applications.
 * Provides better Time-To-First-Token (TTFT) for individual requests by
 * processing new requests immediately without waiting for decode batches.
 */
class PrefillFirstScheduler : public Scheduler {
 public:
  using Scheduler::Scheduler;

 protected:
  bool shouldPrefillFirst(size_t /*decode_count*/,
                          size_t /*max_in_flight_count*/) const override {
    return true;
  }
};

}  // namespace tt::runners::schedulers
