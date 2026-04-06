// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "services/stop_string_processor.hpp"

#include <stdexcept>

#include "utils/logger.hpp"

namespace tt::services {

StopStringProcessor::StopStringProcessor(CancelCallback cancel_callback)
    : cancel_callback_(std::move(cancel_callback)) {
  if (!cancel_callback_) {
    throw std::invalid_argument(
        "StopStringProcessor: cancel_callback must not be null");
  }
}

void StopStringProcessor::initializeTask(
    uint32_t task_id, const std::vector<std::string>& stop_sequences) {
  std::lock_guard<std::mutex> lock(mutex_);

  TaskState state;
  state.stop_sequences = stop_sequences;
  state.stop_detected = false;
  state.accumulated_text.clear();

  task_states_[task_id] = std::move(state);

  TT_LOG_DEBUG(
      "[StopStringProcessor] Initialized task {} with {} stop sequences",
      task_id, stop_sequences.size());
}

StopCheckResult StopStringProcessor::processText(uint32_t task_id,
                                                 const std::string& text) {
  std::unique_lock<std::mutex> lock(mutex_);

  // Find task state
  auto it = task_states_.find(task_id);
  if (it == task_states_.end()) {
    TT_LOG_WARN(
        "[StopStringProcessor] processText called for uninitialized task {}",
        task_id);
    return {false, "", text};
  }

  TaskState& state = it->second;

  // If stop already detected, return empty (skip accumulation)
  if (state.stop_detected) {
    return {true, "", ""};
  }

  // Accumulate text
  state.accumulated_text += text;

  // Check each stop sequence for suffix match
  for (const auto& stop_seq : state.stop_sequences) {
    if (stop_seq.empty()) {
      continue;
    }

    const std::string& acc_text = state.accumulated_text;
    if (acc_text.size() >= stop_seq.size() &&
        acc_text.compare(acc_text.size() - stop_seq.size(), stop_seq.size(),
                         stop_seq) == 0) {
      // Found a match!
      state.stop_detected = true;

      // Truncate stop string from accumulated text
      state.accumulated_text.resize(acc_text.size() - stop_seq.size());

      TT_LOG_INFO("[StopStringProcessor] Stop string '{}' detected for task {}",
                  stop_seq, task_id);

      // CRITICAL: Unlock mutex before invoking callback to avoid deadlock
      lock.unlock();

      // Invoke cancel callback to broadcast abort
      cancel_callback_(task_id);

      // Re-acquire mutex for return
      lock.lock();

      return {true, stop_seq, state.accumulated_text};
    }
  }

  // No match found
  return {false, "", text};
}

bool StopStringProcessor::isStopDetected(uint32_t task_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = task_states_.find(task_id);
  if (it == task_states_.end()) {
    return false;
  }

  return it->second.stop_detected;
}

void StopStringProcessor::finalizeTask(uint32_t task_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = task_states_.find(task_id);
  if (it == task_states_.end()) {
    TT_LOG_WARN(
        "[StopStringProcessor] finalizeTask called for non-existent task {}",
        task_id);
    return;
  }

  task_states_.erase(it);

  TT_LOG_DEBUG("[StopStringProcessor] Finalized task {}", task_id);
}

size_t StopStringProcessor::activeTaskCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return task_states_.size();
}

}  // namespace tt::services
