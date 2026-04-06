// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::services {

/**
 * Result of stop string checking for a single processText call.
 */
struct StopCheckResult {
  bool stop_detected = false;
  std::string matched_string;
  std::string output_text;  // Text with stop sequence truncated if matched
};

/**
 * Processor for detecting stop strings in streaming LLM output.
 *
 * This class manages per-task state for accumulating text and checking
 * for stop sequences. When a stop sequence is detected, it:
 * - Truncates the matched stop string from the output
 * - Invokes a cancel callback to abort the generation
 * - Marks the task as stopped for subsequent calls
 *
 * Thread-safe for concurrent access from multiple consumer threads.
 * Follows the ReasoningParser pattern for lifecycle management.
 */
class StopStringProcessor {
 public:
  using CancelCallback = std::function<void(uint32_t taskId)>;

  /**
   * Constructs a StopStringProcessor with a cancel callback.
   *
   * @param cancel_callback Function to invoke when stop string is detected.
   *        Called with the task ID. Must not be null.
   * @throws std::invalid_argument if cancel_callback is null
   */
  explicit StopStringProcessor(CancelCallback cancel_callback);

  ~StopStringProcessor() = default;

  StopStringProcessor(const StopStringProcessor&) = delete;
  StopStringProcessor& operator=(const StopStringProcessor&) = delete;

  /**
   * Initialize stop string checking for a task.
   *
   * Creates internal state for the task with the given stop sequences.
   * Should be called once per task before any processText calls.
   *
   * @param task_id Unique task identifier
   * @param stop_sequences List of strings to detect (can be empty)
   */
  void initializeTask(uint32_t task_id,
                      const std::vector<std::string>& stop_sequences);

  /**
   * Process a chunk of text for stop string detection.
   *
   * Accumulates the text and checks if the accumulated text ends with any
   * stop sequence. If a match is found:
   * - Sets stop_detected flag for this task
   * - Truncates the matched stop string from accumulated text
   * - Invokes cancel_callback (with mutex unlocked to avoid deadlock)
   *
   * Subsequent calls after stop detected will return empty output.
   *
   * @param task_id Task identifier (must be initialized)
   * @param text Text chunk to process
   * @return StopCheckResult with detection status and processed text
   */
  StopCheckResult processText(uint32_t task_id, const std::string& text);

  /**
   * Check if stop has been detected for a task.
   *
   * @param task_id Task identifier
   * @return true if stop detected, false otherwise (or if task not found)
   */
  bool isStopDetected(uint32_t task_id) const;

  /**
   * Finalize stop string checking for a task.
   *
   * Removes internal state for the task. Should be called once per task
   * when generation is complete (success, error, or abort).
   *
   * @param task_id Task identifier
   */
  void finalizeTask(uint32_t task_id);

  /**
   * Get count of active tasks being tracked.
   *
   * @return Number of tasks with initialized state
   */
  size_t activeTaskCount() const;

 private:
  struct TaskState {
    std::string accumulated_text;
    std::vector<std::string> stop_sequences;
    bool stop_detected = false;
  };

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, TaskState> task_states_;
  CancelCallback cancel_callback_;
};

}  // namespace tt::services
