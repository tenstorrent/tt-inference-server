// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include "runners/runner_base.hpp"

namespace tt::runners {

/** Loop-driven runner used by the IPC worker process (LLM, Embedding, ...).
 *  The worker calls `start()`, which warms up and then enters `run()`. */
class IRunner : public IRunnerBase {
 public:
  /** Calls `onWarmupDone` once warmup() returns true, then enters `run()`. */
  void start(std::function<void()> onWarmupDone = nullptr) {
    if (!warmup()) {
      throw std::runtime_error(std::string(runnerType()) + " warmup failed");
    }
    if (onWarmupDone) {
      onWarmupDone();
    }
    run();
  }

 private:
  virtual void run() = 0;
};

}  // namespace tt::runners
