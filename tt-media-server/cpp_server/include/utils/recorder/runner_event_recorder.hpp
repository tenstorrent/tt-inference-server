// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tt::domain {
class Sequence;
}

namespace tt::utils::recorder {

/**
 * Modes:
 *  - OFF    -- no-op (default; production)
 *  - RECORD -- append events as JSONL to TT_RUNNER_RECORDER_PATH
 *  - ASSERT -- compare events emitted at runtime against the JSONL fixture
 *              at TT_RUNNER_RECORDER_PATH; mismatches are logged and cause
 *              finalize() to return false.
 *
 * Activation is driven by env vars (read once at first access):
 *   TT_RUNNER_RECORDER_MODE = off | record | assert
 *   TT_RUNNER_RECORDER_PATH = path to JSONL file
 *
 * The recorder lives in the main server process (where API/Service code
 * runs). It is instrumented at the producer side of the task / cancel
 * queues, capturing exactly what the API+Service layer hands to the
 * worker/runner. This makes it stable across refactors of API,
 * controllers, services, session manager, and disaggregation glue.
 *
 * Thread-safety: a single mutex serializes record/assert events. The set
 * of API request submissions is sequential per scenario script, so total
 * order is well-defined.
 */
enum class Mode { OFF, RECORD, ASSERT };

class RunnerEventRecorder {
 public:
  static RunnerEventRecorder& instance();

  Mode mode() const { return mode_; }
  bool isActive() const { return mode_ != Mode::OFF; }

  /** Called from LLMService::processStreamingRequest just before push. */
  void onTaskSubmitted(const tt::domain::Sequence& sequence);

  /** Called from LLMService::abortRequest just before broadcasting cancel. */
  void onCancelRequested(uint32_t taskId);

  /**
   * Flush the record file (RECORD) or verify all expected events have been
   * consumed (ASSERT). Returns true on success.
   * Safe to call multiple times.
   */
  bool finalize();

  /** Mismatch counter, useful for tests. */
  size_t mismatchCount() const { return mismatchCount_.load(); }

 private:
  RunnerEventRecorder();
  ~RunnerEventRecorder();

  RunnerEventRecorder(const RunnerEventRecorder&) = delete;
  RunnerEventRecorder& operator=(const RunnerEventRecorder&) = delete;

  void emit(const std::string& jsonLine);

  Mode mode_ = Mode::OFF;
  std::string path_;

  std::mutex mutex_;
  std::ofstream out_;

  std::vector<std::string> expected_;
  size_t expectedCursor_ = 0;
  std::atomic<size_t> mismatchCount_{0};
  std::atomic<bool> finalized_{false};
};

}  // namespace tt::utils::recorder
