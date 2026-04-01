// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/reasoning_parser.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "utils/logger.hpp"

namespace tt::services {

ReasoningParseResult ReasoningParser::parseComplete(
    const std::string& text) const {
  ReasoningParseResult result;

  // Check if text starts with <think>
  size_t startLen = std::strlen(THINK_START_TAG);
  size_t endLen = std::strlen(THINK_END_TAG);

  if (text.size() < startLen ||
      text.compare(0, startLen, THINK_START_TAG) != 0) {
    // No <think> tag at start - treat entire text as answer
    result.reasoning = std::nullopt;
    result.answer = text;
    result.has_reasoning = false;
    result.is_malformed = false;
    return result;
  }

  // Has <think> tag at start
  result.has_reasoning = true;

  // Find </think> delimiter
  size_t endPos = text.find(THINK_END_TAG, startLen);

  if (endPos == std::string::npos) {
    // Malformed: has <think> but no </think>
    // Treat everything after <think> as reasoning, no answer
    std::string reasoningContent = text.substr(startLen);

    // Trim leading/trailing whitespace from reasoning
    size_t first = reasoningContent.find_first_not_of("\n\r\t ");
    size_t last = reasoningContent.find_last_not_of("\n\r\t ");

    if (first != std::string::npos && last != std::string::npos) {
      result.reasoning = reasoningContent.substr(first, last - first + 1);
    } else {
      result.reasoning = reasoningContent;
    }

    result.answer = "";
    result.is_malformed = true;

    TT_LOG_WARN(
        "[ReasoningParser] Malformed output: found <think> but no </think>");
    return result;
  }

  // Extract reasoning (between <think> and </think>)
  std::string reasoningContent = text.substr(startLen, endPos - startLen);

  // Trim leading/trailing whitespace from reasoning
  size_t first = reasoningContent.find_first_not_of("\n\r\t ");
  size_t last = reasoningContent.find_last_not_of("\n\r\t ");

  if (first != std::string::npos && last != std::string::npos) {
    result.reasoning = reasoningContent.substr(first, last - first + 1);
  } else {
    result.reasoning = reasoningContent;
  }

  // Extract answer (after </think>)
  size_t answerStart = endPos + endLen;
  std::string answerContent = text.substr(answerStart);

  // Trim leading whitespace from answer
  first = answerContent.find_first_not_of("\n\r\t ");
  if (first != std::string::npos) {
    result.answer = answerContent.substr(first);
  } else {
    result.answer = answerContent;
  }

  result.is_malformed = false;
  return result;
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

  if (tokenId == THINK_START_TOKEN) {
    // Entering reasoning block
    state.in_reasoning = true;
    state.seen_think_start = true;
    TT_LOG_DEBUG("[ReasoningParser] Task {} entered reasoning block", taskId);
    return {ContentType::REASONING, "", false};

  } else if (tokenId == THINK_END_TOKEN) {
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
