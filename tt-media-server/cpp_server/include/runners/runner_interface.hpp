// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>
#include <stdexcept>

namespace tt::runners {

/**
 * Common interface for all runners (LLM, Embedding, etc.).
 * Provides basic lifecycle management for inference runners.
 */
class IRunner {
 public:
  virtual ~IRunner() = default;

  /**
   * Stop the runner gracefully.
   */
  virtual void stop() = 0;

  /**
   * Warm up the runner by preloading models and resources.
   */
  bool warmup() { return true; }

  void start() {
    // Initialize resources and prepare for inference.
    bool isWarmedUp = warmup();
    if (!isWarmedUp) {
      throw std::runtime_error(std::string(runnerType()) + " warmup failed");
    }
    run();
  }

  /**
   * Get the runner type for identification.
   */
  virtual const char* runnerType() const = 0;

 private:
  /**
   * Start the runner and begin processing.
   * This method should run the main inference loop.
   */
  virtual void run() = 0;
};

}  // namespace tt::runners
