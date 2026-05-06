// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include "domain/llm/sequence.hpp"

namespace tt::ipc {

using namespace tt::domain::llm;

/**
 * Abstract interface for the scheduler's task queue.
 *
 * - push   --  enqueues a sequence (may block if queue is full).
 * - try_pop -- non-blocking pop; returns nullptr when empty, otherwise a
 *              unique_ptr<Sequence> (caller owns the pointer).
 * - empty  -- emptiness check.
 */
class ITaskQueue {
 public:
  virtual ~ITaskQueue() = default;
  virtual void push(const tt::domain::llm::Sequence& seq) = 0;
  virtual std::unique_ptr<tt::domain::llm::Sequence>
  tryPop() = 0;  // non-blocking pop;
  virtual std::unique_ptr<tt::domain::llm::Sequence>
  receive() = 0;  // blocking pop;
  virtual bool empty() const = 0;
};

}  // namespace tt::ipc
