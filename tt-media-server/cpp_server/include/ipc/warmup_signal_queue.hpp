// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

namespace tt::ipc {

/**
 * Abstract interface for worker warmup signaling (worker → main).
 * Allows swapping implementation (e.g. Boost message_queue, socket) without
 * changing call sites.
 */
class IWarmupSignalQueue {
 public:
  virtual ~IWarmupSignalQueue() = default;

  /** Worker calls after runner warmup (open_only usage). */
  virtual void sendReady(int workerId) = 0;

  /** Main process blocks until one worker signals (create_only usage). Returns
   * worker id. */
  virtual int receive() = 0;

  /** Remove the named queue (main process only, before destructor). Default
   * no-op. */
  virtual void remove() {}
};

}  // namespace tt::ipc
