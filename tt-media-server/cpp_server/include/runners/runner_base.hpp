// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::runners {

/**
 * Lifecycle base shared by both runner families. `IRunner` adds a no-arg
 * `run()` loop for IPC workers; `MediaRunner<R, S>` adds a synchronous
 * `S run(const R&)` for in-process services.
 */
class IRunnerBase {
 public:
  virtual ~IRunnerBase() = default;

  virtual bool warmup() { return true; }
  virtual void stop() {}
  virtual const char* runnerType() const = 0;
};

}  // namespace tt::runners
