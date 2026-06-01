// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/reasoning_parser.hpp"

#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::services {

ReasoningParser::ReasoningParser() {
  const auto [thinkStart, thinkEnd] = tt::utils::tokenizers::thinkTokenIds();
  thinkStartToken_ = thinkStart;
  thinkEndToken_ = thinkEnd;
  thinkTokensEnabled_ = thinkStart != tt::utils::tokenizers::kNoThinkTokenId &&
                        thinkEnd != tt::utils::tokenizers::kNoThinkTokenId;
}

void ReasoningParser::initializeTask(uint32_t taskId) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Initialize or reset task state
  TaskState& state = task_states_[taskId];
  state.in_reasoning = false;
  state.seen_think_start = false;

  TT_LOG_DEBUG("[ReasoningParser] Initialized task: {}", taskId);
}

TokenParseResult ReasoningParser::processToken(uint32_t taskId, int64_t tokenId,
                                               const std::string& decodedText) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = task_states_.find(taskId);
  if (it == task_states_.end()) {
    // Task not initialized - treat as normal answer content
    TT_LOG_WARN(
        "[ReasoningParser] processToken called for uninitialized task: {}",
        taskId);
    return {ContentType::ANSWER, decodedText, true};
  }

  TaskState& state = it->second;

  if (thinkTokensEnabled_ && tokenId == thinkStartToken_) {
    // Entering reasoning block
    state.in_reasoning = true;
    state.seen_think_start = true;
    TT_LOG_DEBUG("[ReasoningParser] Task {} entered reasoning block", taskId);
    return {ContentType::REASONING, "", false};

  } else if (thinkTokensEnabled_ && tokenId == thinkEndToken_) {
    // Exiting reasoning block
    if (!state.in_reasoning) {
      TT_LOG_WARN(
          "[ReasoningParser] Task {} found </think> without matching <think>",
          taskId);
    }
    state.in_reasoning = false;
    TT_LOG_DEBUG("[ReasoningParser] Task {} exited reasoning block", taskId);
    return {ContentType::REASONING, "", false};

  } else if (state.in_reasoning) {
    // Inside reasoning block - tag as reasoning content
    return {ContentType::REASONING, decodedText, true};

  } else {
    // Outside reasoning block - tag as normal answer content
    return {ContentType::ANSWER, decodedText, true};
  }
}

void ReasoningParser::finalizeTask(uint32_t taskId) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = task_states_.find(taskId);
  if (it == task_states_.end()) {
    TT_LOG_WARN("[ReasoningParser] finalizeTask called for unknown task: {}",
                taskId);
    return;
  }

  const TaskState& state = it->second;

  // Warn if task ended while still in reasoning block
  if (state.in_reasoning || (state.seen_think_start && state.in_reasoning)) {
    TT_LOG_WARN(
        "[ReasoningParser] Task {} ended while still in reasoning block "
        "(incomplete reasoning)",
        taskId);
  }

  task_states_.erase(it);
  TT_LOG_DEBUG("[ReasoningParser] Finalized task: {}", taskId);
}

bool ReasoningParser::isInReasoning(uint32_t taskId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = task_states_.find(taskId);
  if (it == task_states_.end()) {
    return false;
  }

  return it->second.in_reasoning;
}

size_t ReasoningParser::activeTaskCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return task_states_.size();
}

}  // namespace tt::services
