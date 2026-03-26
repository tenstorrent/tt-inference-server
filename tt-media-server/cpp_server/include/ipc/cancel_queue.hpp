// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <vector>

#include "domain/task_id.hpp"

namespace tt::ipc {

/**
 * Abstract interface for request-cancel signaling (main → worker).
 * Allows swapping implementation (e.g. Boost message_queue, shared memory)
 * without changing call sites.
 */
class ICancelQueue {
 public:
  virtual ~ICancelQueue() = default;

  /**
   * Non-blocking push. Implementations should silently drop the message if
   * the queue is full (the request will finish on its own shortly anyway).
   * Thread-safe.
   */
  virtual void push(const tt::domain::TaskID& taskId) = 0;

  /**
   * Drain all available cancel messages into out without blocking.
   * Intended to be called once per scheduler step from the worker thread.
   */
  virtual void tryPopAll(std::vector<tt::domain::TaskID>& out) = 0;

  /** Remove the underlying queue resource. Default no-op. */
  virtual void remove() {}
};

}  // namespace tt::ipc
