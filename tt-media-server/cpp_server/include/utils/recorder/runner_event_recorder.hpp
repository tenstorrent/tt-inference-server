// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace tt::domain {
class Sequence;
}

namespace tt::utils::recorder {

/**
 * In-memory event log that captures everything the API/Service layer hands
 * to the worker queue (Sequences submitted to the task queue, cancel
 * signals broadcast to cancel queues).
 *
 * The recorder is hooked at the producer side of the queues from
 * LLMService, so it lives in the main server process and observes events
 * regardless of which runner the worker uses.
 *
 * Activation: opt-in via env var TT_RUNNER_RECORDER_ENABLED=1. When
 * disabled, the hook calls compile down to atomic-load + branch and the
 * buffer stays empty.
 *
 * Tests do not read fixtures from disk -- a debug HTTP endpoint
 * (`GET /debug/runner-events`, `DELETE /debug/runner-events`) exposes the
 * buffer so test code can assert inline:
 *
 *     DELETE /debug/runner-events
 *     POST   /v1/chat/completions  ...
 *     GET    /debug/runner-events  -> assert what was submitted
 *
 * Thread safety: snapshots and writes are serialized by an internal mutex.
 * The buffer is bounded; oldest events are dropped with a logged warning
 * once MAX_EVENTS is reached.
 */
class RunnerEventRecorder {
 public:
  static constexpr size_t MAX_EVENTS = 4096;

  struct Event {
    uint64_t seq = 0;
    std::string json;  // self-contained JSON object, no trailing newline
  };

  static RunnerEventRecorder& instance();

  bool isEnabled() const { return enabled_.load(std::memory_order_acquire); }

  /** Producer-side hooks (called from LLMService). */
  void onTaskSubmitted(const tt::domain::Sequence& sequence);
  void onCancelRequested(uint32_t taskId);

  /** Reader API for the debug HTTP controller. */
  std::vector<Event> snapshot(uint64_t sinceSeq = 0) const;
  void clear();
  uint64_t lastSeq() const {
    return seqCounter_.load(std::memory_order_acquire);
  }
  size_t droppedCount() const {
    return droppedCount_.load(std::memory_order_acquire);
  }

 private:
  RunnerEventRecorder();
  ~RunnerEventRecorder() = default;

  RunnerEventRecorder(const RunnerEventRecorder&) = delete;
  RunnerEventRecorder& operator=(const RunnerEventRecorder&) = delete;

  void append(std::string json);

  std::atomic<bool> enabled_{false};
  std::atomic<uint64_t> seqCounter_{0};
  std::atomic<size_t> droppedCount_{0};

  mutable std::mutex mutex_;
  std::deque<Event> events_;
};

}  // namespace tt::utils::recorder
