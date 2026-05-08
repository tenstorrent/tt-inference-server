// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::runners {

/**
 * Lifecycle interface shared by every runner (IPC-loop and direct-call alike).
 * Concrete `run()` shapes belong to the derived families: `IRunner` exposes a
 * no-arg loop driven by the worker process, while `MediaRunner<R, S>` exposes
 * a synchronous `S run(const R&)` invoked directly by its service.
 */
class IRunnerBase {
 public:
  virtual ~IRunnerBase() = default;

  virtual bool warmup() { return true; }
  virtual void stop() {}
  virtual const char* runnerType() const = 0;
};

}  // namespace tt::runners
