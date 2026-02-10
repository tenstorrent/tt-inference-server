// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

/**
 * Abstract interface for the scheduler's task queue.
 *
 * - push   -- serializes and enqueues a sequence (may block if queue is full).
 * - try_pop -- non-blocking pop; returns nullptr when empty, otherwise a
 *              heap-allocated Sequence* (caller owns the pointer).
 * - empty  -- approximate emptiness check (sufficient for is_finished()).
 */
class ITaskQueue {
 public:
  virtual ~ITaskQueue() = default;
  virtual void push(const Sequence& seq) = 0;
  virtual Sequence* try_pop() = 0;
  virtual bool empty() const = 0;
};

}  // namespace llm_engine
