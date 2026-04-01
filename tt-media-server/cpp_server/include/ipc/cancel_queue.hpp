// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <vector>

#include "domain/task_id.hpp"

namespace tt::ipc {

/**
 * Abstract interface for a cancel-signal queue (main process -> worker).
 *
 * - push      -- non-blocking enqueue of a cancel signal.
 * - tryPopAll -- drain all pending cancel signals.
 */
class ICancelQueue {
 public:
  virtual ~ICancelQueue() = default;

  /** Non-blocking push. Drops the message with a warning if the queue is full.
   */
  virtual void push(const domain::TaskID& taskId) = 0;

  /** Drain all available cancel messages into @p out. Non-blocking. */
  virtual void tryPopAll(std::vector<domain::TaskID>& out) = 0;

  /** Remove the underlying IPC resource. Default no-op. */
  virtual void remove() {}
};

}  // namespace tt::ipc
