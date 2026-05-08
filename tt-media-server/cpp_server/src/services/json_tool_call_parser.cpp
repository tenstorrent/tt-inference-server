// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <json/reader.h>
#include <json/value.h>

#include <mutex>
#include <unordered_map>

#include "services/tool_call_parser.hpp"
#include "utils/logger.hpp"
#include "utils/tool_call_id_generator.hpp"

namespace tt::services {

namespace {

// State for structured output streaming - filters out {"arguments": wrapper
enum class StructuredOutputState {
  SKIPPING_PREFIX,  // Accumulating and matching {"arguments":
  STREAMING,        // Streaming the actual arguments content
  DONE              // Finished (skipping trailing wrapper)
};

struct StructuredOutputParseState {
  StructuredOutputState state = StructuredOutputState::SKIPPING_PREFIX;
  std::string buffer;      // Accumulation buffer for prefix matching
  int braceDepth = 0;      // Track nested braces to know when arguments end
  bool sentStart = false;  // Whether TOOL_CALL_START has been sent
  std::string toolCallId;  // Generated tool call ID
};

// Helper to build tool_calls JSON array for a delta
Json::Value buildToolCallsDelta(int index, ToolCallDeltaType deltaType,
                                const std::string& callId = "",
                                const std::string& functionName = "",
                                const std::string& argumentsDelta = "") {
  Json::Value toolCallsArray(Json::arrayValue);
  Json::Value toolCallDelta;
  toolCallDelta["index"] = index;

  switch (deltaType) {
    case ToolCallDeltaType::TOOL_CALL_START:
      toolCallDelta["id"] = callId;
      toolCallDelta["type"] = "function";
      toolCallDelta["function"]["name"] = functionName;
      toolCallDelta["function"]["arguments"] = "";
      break;

    case ToolCallDeltaType::ARGUMENTS_DELTA:
      toolCallDelta["function"]["arguments"] = argumentsDelta;
      break;

    case ToolCallDeltaType::TOOL_CALL_END:
      // Empty delta for end marker
      break;

    default:
      break;
  }

  toolCallsArray.append(toolCallDelta);
  return toolCallsArray;
}

ToolCallTokenResult makeArgumentsDelta(int index, const std::string& delta) {
  return {ToolCallDeltaType::ARGUMENTS_DELTA,
          index,
          delta,
          "",
          "",
          buildToolCallsDelta(index, ToolCallDeltaType::ARGUMENTS_DELTA, "", "",
                              delta)};
}

/**
 * JSON Tool Call Parser for structured output.
 *
 * Handles models outputting JSON wrapper format like:
 *   {"arguments":{"location":"SF"},"name":"get_weather"}
 *
 * Strips the wrapper and streams only the inner arguments content.
 * Used when tool_choice is "function" or "required".
 */
class JsonToolCallParser : public IToolCallParser {
 public:
  std::string stripMarkers(const std::string& text) const override {
    // No markers to strip for JSON format
    return text;
  }

  void initializeTask(uint32_t taskId) override {
    initializeTask(taskId, "unknown");
  }

  void initializeTask(uint32_t taskId,
                      const std::string& functionName) override {
    std::lock_guard<std::mutex> lock(mutex_);

    JsonTaskState& state = taskStates_[taskId];
    state.parseState = StructuredOutputParseState{};
    state.parseState.toolCallId = tt::utils::ToolCallIDGenerator::generate();
    state.functionName = functionName;
    state.accumulatedArgs.clear();

    TT_LOG_DEBUG("[JsonToolCallParser] Initialized task: {} with function: {}",
                 taskId, functionName);
  }

  std::optional<ToolCallTokenResult> processToken(
      uint32_t taskId, [[maybe_unused]] int64_t tokenId,
      const std::string& decodedText) override {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = taskStates_.find(taskId);
    if (it == taskStates_.end()) {
      TT_LOG_WARN(
          "[JsonToolCallParser] processToken called for uninitialized task: {}",
          taskId);
      return std::nullopt;
    }

    JsonTaskState& state = it->second;
    StructuredOutputParseState& parseState = state.parseState;

    static constexpr std::string_view kArgsMarker = "\"arguments\":";
    std::string filteredDelta;

    for (char c : decodedText) {
      switch (parseState.state) {
        case StructuredOutputState::SKIPPING_PREFIX:
          parseState.buffer += c;
          {
            size_t pos = parseState.buffer.find(kArgsMarker);
            if (pos != std::string::npos) {
              parseState.state = StructuredOutputState::STREAMING;
              parseState.buffer.clear();
              parseState.braceDepth = 0;
            } else if (parseState.buffer.size() > 100) {
              filteredDelta += parseState.buffer;
              parseState.buffer.clear();
              parseState.state = StructuredOutputState::STREAMING;
              parseState.braceDepth = 0;
              for (char bc : filteredDelta) {
                if (bc == '{')
                  parseState.braceDepth++;
                else if (bc == '}')
                  parseState.braceDepth--;
              }
            }
          }
          break;

        case StructuredOutputState::STREAMING:
          if (c == '{') {
            parseState.braceDepth++;
            filteredDelta += c;
          } else if (c == '}') {
            if (parseState.braceDepth > 0) {
              parseState.braceDepth--;
              filteredDelta += c;
            }
            if (parseState.braceDepth == 0) {
              parseState.state = StructuredOutputState::DONE;
            }
          } else {
            filteredDelta += c;
          }
          break;

        case StructuredOutputState::DONE:
          break;
      }
    }

    state.accumulatedArgs += filteredDelta;

    // Emit TOOL_CALL_START on first non-empty filtered delta
    if (!filteredDelta.empty() && !parseState.sentStart) {
      parseState.sentStart = true;
      TT_LOG_DEBUG(
          "[JsonToolCallParser] Task {} emitting TOOL_CALL_START: function={}",
          taskId, state.functionName);
      return ToolCallTokenResult{
          ToolCallDeltaType::TOOL_CALL_START,
          0,
          "",
          state.functionName,
          parseState.toolCallId,
          buildToolCallsDelta(0, ToolCallDeltaType::TOOL_CALL_START,
                              parseState.toolCallId, state.functionName)};
    }

    // Emit arguments delta
    if (!filteredDelta.empty()) {
      return makeArgumentsDelta(0, filteredDelta);
    }

    // No content to emit yet
    return std::nullopt;
  }

  std::optional<Json::Value> finalizeTask(uint32_t taskId) override {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = taskStates_.find(taskId);
    if (it == taskStates_.end()) {
      TT_LOG_WARN(
          "[JsonToolCallParser] finalizeTask called for unknown task: {}",
          taskId);
      return std::nullopt;
    }

    JsonTaskState& state = it->second;

    // Build final tool call result
    std::optional<Json::Value> result;
    if (!state.accumulatedArgs.empty()) {
      Json::Value toolCallsArray(Json::arrayValue);
      Json::Value toolCall;
      toolCall["id"] = state.parseState.toolCallId;
      toolCall["type"] = "function";
      toolCall["function"]["name"] = state.functionName;
      toolCall["function"]["arguments"] = state.accumulatedArgs;
      toolCallsArray.append(toolCall);
      result = toolCallsArray;

      TT_LOG_DEBUG(
          "[JsonToolCallParser] Task {} finalized with tool call: function={}",
          taskId, state.functionName);
    }

    taskStates_.erase(it);
    return result;
  }

  bool isInToolCall(uint32_t taskId) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = taskStates_.find(taskId);
    return it != taskStates_.end();
  }

  size_t activeTaskCount() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return taskStates_.size();
  }

 private:
  struct JsonTaskState {
    StructuredOutputParseState parseState;
    std::string functionName;
    std::string accumulatedArgs;
  };

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, JsonTaskState> taskStates_;
};

}  // namespace

std::unique_ptr<IToolCallParser> createJsonToolCallParser() {
  return std::make_unique<JsonToolCallParser>();
}

}  // namespace tt::services
