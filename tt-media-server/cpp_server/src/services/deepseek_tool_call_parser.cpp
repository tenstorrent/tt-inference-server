// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <json/reader.h>
#include <json/value.h>

#include <mutex>
#include <regex>
#include <unordered_map>

#include "config/types.hpp"
#include "services/tool_call_parser.hpp"
#include "utils/logger.hpp"
#include "utils/tool_call_id_generator.hpp"

namespace tt::services {

namespace {

// Token IDs for DeepSeek tool call markers
constexpr int64_t TOOL_CALLS_BEGIN_TOKEN = 128806;  // <｜tool▁calls▁begin｜>
constexpr int64_t TOOL_CALLS_END_TOKEN = 128807;    // <｜tool▁calls▁end｜>
constexpr int64_t TOOL_CALL_BEGIN_TOKEN = 128808;   // <｜tool▁call▁begin｜>
constexpr int64_t TOOL_CALL_END_TOKEN = 128809;     // <｜tool▁call▁end｜>
constexpr int64_t TOOL_SEP_TOKEN = 128814;          // <｜tool▁sep｜>

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

// Parsing state for streaming tool calls
enum class ParsingState {
  REGULAR,           // Outside tool calls
  IN_TOOL_CALLS,     // Inside tool calls block
  IN_TOOL_CALL,      // Inside individual tool call, before separator
  IN_FUNCTION_NAME,  // Parsing function name after separator
  IN_ARGUMENTS       // Parsing JSON arguments
};

// Per-task state for streaming tool call parsing
struct ToolCallTaskState {
  ParsingState state = ParsingState::REGULAR;
  std::string buffer;             // Accumulation buffer
  std::string current_function;   // Current function name being parsed
  std::string current_arguments;  // Current arguments being parsed
  Json::Value tool_calls;         // Array of completed tool calls
  int call_index = 0;             // Tool call counter
  bool in_json_block = false;     // Inside ```json...``` block
  bool sent_tool_call_start =
      false;  // Whether we've sent TOOL_CALL_START for current call
  std::string current_tool_call_id;  // ID for current tool call being parsed
  std::string arguments_buffer;      // Buffer for arguments being streamed
};

/**
 * DeepSeek tool call format parser.
 *
 * Expected format:
 * <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
 * ```json
 * {"location":"San Francisco"}
 * ```
 * <｜tool▁call▁end｜><｜tool▁calls▁end｜>
 */
class DeepSeekToolCallParser : public IToolCallParser {
 public:
  void initializeTask(uint32_t taskId) override {
    std::lock_guard<std::mutex> lock(mutex);

    // Initialize or reset task state
    ToolCallTaskState& state = taskStates[taskId];
    state.state = ParsingState::REGULAR;
    state.buffer.clear();
    state.current_function.clear();
    state.current_arguments.clear();
    state.tool_calls = Json::Value(Json::arrayValue);
    state.call_index = 0;
    state.in_json_block = false;
    state.sent_tool_call_start = false;
    state.current_tool_call_id.clear();
    state.arguments_buffer.clear();

    TT_LOG_DEBUG("[ToolCallParser] Initialized task: {}", taskId);
  }

  std::optional<ToolCallTokenResult> processToken(
      uint32_t taskId, int64_t tokenId,
      const std::string& decodedText) override {
    std::lock_guard<std::mutex> lock(mutex);

    TT_LOG_DEBUG(
        "[ToolCallParser] processToken called for task: {}, tokenId: {}, "
        "decodedText: {}",
        taskId, tokenId, decodedText);
    auto it = taskStates.find(taskId);
    if (it == taskStates.end()) {
      TT_LOG_WARN(
          "[ToolCallParser] processToken called for uninitialized task: {}",
          taskId);
      return std::nullopt;  // Not in tool call mode, handle as regular
    }

    ToolCallTaskState& state = it->second;

    // Handle special token markers
    if (tokenId == TOOL_CALLS_BEGIN_TOKEN) {
      state.state = ParsingState::IN_TOOL_CALLS;
      TT_LOG_DEBUG("[ToolCallParser] Task {} entered tool calls block", taskId);
      return std::nullopt;

    } else if (tokenId == TOOL_CALLS_END_TOKEN) {
      state.state = ParsingState::REGULAR;
      TT_LOG_DEBUG("[ToolCallParser] Task {} exited tool calls block", taskId);
      return std::nullopt;

    } else if (tokenId == TOOL_CALL_BEGIN_TOKEN) {
      state.state = ParsingState::IN_TOOL_CALL;
      state.buffer.clear();
      state.current_function.clear();
      state.current_arguments.clear();
      state.arguments_buffer.clear();
      state.in_json_block = false;
      state.sent_tool_call_start = false;
      state.current_tool_call_id = tt::utils::ToolCallIDGenerator::generate();
      TT_LOG_DEBUG("[ToolCallParser] Task {} started new tool call, id={}",
                   taskId, state.current_tool_call_id);
      return std::nullopt;

    } else if (tokenId == TOOL_CALL_END_TOKEN) {
      finalizeSingleToolCall(state);
      state.state = ParsingState::IN_TOOL_CALLS;
      int idx = state.call_index - 1;
      TT_LOG_DEBUG("[ToolCallParser] Task {} ended tool call index {}", taskId,
                   idx);
      return ToolCallTokenResult{
          ToolCallDeltaType::TOOL_CALL_END,
          idx,
          "",
          "",
          "",
          buildToolCallsDelta(idx, ToolCallDeltaType::TOOL_CALL_END)};

    } else if (tokenId == TOOL_SEP_TOKEN) {
      state.buffer.clear();
      state.state = ParsingState::IN_FUNCTION_NAME;
      TT_LOG_DEBUG("[ToolCallParser] Task {} parsing function name", taskId);
      return std::nullopt;
    }

    // Process content based on current state
    switch (state.state) {
      case ParsingState::REGULAR:
        return std::nullopt;  // Not in tool call, handle as regular

      case ParsingState::IN_TOOL_CALLS:
      case ParsingState::IN_TOOL_CALL:
        state.buffer += decodedText;
        return std::nullopt;  // Consumed, suppress

      case ParsingState::IN_FUNCTION_NAME: {
        state.buffer += decodedText;

        if (decodedText.find('\n') != std::string::npos ||
            decodedText.find('`') != std::string::npos) {
          size_t nameEnd = state.buffer.find_first_of("\n`");
          if (nameEnd != std::string::npos) {
            state.current_function = state.buffer.substr(0, nameEnd);
            state.current_function.erase(
                0, state.current_function.find_first_not_of(" \t\n\r"));
            state.current_function.erase(
                state.current_function.find_last_not_of(" \t\n\r") + 1);

            state.buffer.clear();
            state.state = ParsingState::IN_ARGUMENTS;
            state.sent_tool_call_start = true;

            TT_LOG_DEBUG(
                "[ToolCallParser] Task {} emitting TOOL_CALL_START: "
                "function={}, id={}",
                taskId, state.current_function, state.current_tool_call_id);

            int idx = state.call_index;
            return ToolCallTokenResult{
                ToolCallDeltaType::TOOL_CALL_START,
                idx,
                "",
                state.current_function,
                state.current_tool_call_id,
                buildToolCallsDelta(idx, ToolCallDeltaType::TOOL_CALL_START,
                                    state.current_tool_call_id,
                                    state.current_function)};
          }
        }
        return std::nullopt;
      }

      case ParsingState::IN_ARGUMENTS: {
        state.buffer += decodedText;

        if (!state.in_json_block) {
          size_t backtickPos = state.buffer.find("```");
          if (backtickPos != std::string::npos) {
            state.in_json_block = true;
            size_t jsonStart = backtickPos + 3;
            if (state.buffer.substr(jsonStart, 4) == "json") {
              jsonStart += 4;
            }
            while (jsonStart < state.buffer.size() &&
                   (state.buffer[jsonStart] == ' ' ||
                    state.buffer[jsonStart] == '\n' ||
                    state.buffer[jsonStart] == '\t' ||
                    state.buffer[jsonStart] == '\r')) {
              jsonStart++;
            }
            state.buffer = state.buffer.substr(jsonStart);
            state.arguments_buffer.clear();

            if (!state.buffer.empty() &&
                state.buffer.find("```") == std::string::npos) {
              state.arguments_buffer += state.buffer;
              state.current_arguments += state.buffer;
              return makeArgumentsDelta(state.call_index, state.buffer);
            }
          }
          return std::nullopt;

        } else {
          size_t backtickPos = state.buffer.find("```");
          if (backtickPos != std::string::npos) {
            std::string finalContent = state.buffer.substr(0, backtickPos);
            state.current_arguments += finalContent;
            state.buffer.clear();
            state.in_json_block = false;
            TT_LOG_DEBUG("[ToolCallParser] Completed JSON arguments");

            if (!finalContent.empty()) {
              std::string newContent =
                  finalContent.substr(state.arguments_buffer.size());
              state.arguments_buffer = finalContent;
              if (!newContent.empty()) {
                return makeArgumentsDelta(state.call_index, newContent);
              }
            }
            return std::nullopt;
          }

          std::string newContent =
              state.buffer.substr(state.arguments_buffer.size());
          state.arguments_buffer = state.buffer;
          state.current_arguments += newContent;

          if (!newContent.empty()) {
            TT_LOG_DEBUG("[ToolCallParser] Emitting arguments delta: '{}'",
                         newContent);
            return makeArgumentsDelta(state.call_index, newContent);
          }
          return std::nullopt;
        }
      }

      default:
        return std::nullopt;
    }
  }

  std::optional<Json::Value> finalizeTask(uint32_t taskId) override {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = taskStates.find(taskId);
    if (it == taskStates.end()) {
      TT_LOG_WARN("[ToolCallParser] finalizeTask called for unknown task: {}",
                  taskId);
      return std::nullopt;
    }

    ToolCallTaskState& state = it->second;

    // Warn if task ended while still parsing tool call
    if (state.state != ParsingState::REGULAR) {
      TT_LOG_WARN(
          "[ToolCallParser] Task {} ended while still in tool call (state: {})",
          taskId, static_cast<int>(state.state));
    }

    // Return tool calls if any were parsed
    std::optional<Json::Value> result;
    if (state.tool_calls.size() > 0) {
      result = state.tool_calls;
      TT_LOG_DEBUG("[ToolCallParser] Task {} finalized with {} tool calls",
                   taskId, state.tool_calls.size());
    }

    taskStates.erase(it);
    TT_LOG_DEBUG("[ToolCallParser] Finalized task: {}", taskId);

    return result;
  }

  bool isInToolCall(uint32_t taskId) const override {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = taskStates.find(taskId);
    if (it == taskStates.end()) {
      return false;
    }

    return it->second.state != ParsingState::REGULAR;
  }

  size_t activeTaskCount() const override {
    std::lock_guard<std::mutex> lock(mutex);
    return taskStates.size();
  }

 private:
  mutable std::mutex mutex;
  std::unordered_map<uint32_t, ToolCallTaskState> taskStates;

  // Helper to finalize a single tool call and add it to the array
  void finalizeSingleToolCall(ToolCallTaskState& state) {
    if (state.current_function.empty()) {
      TT_LOG_WARN(
          "[ToolCallParser] Finalizing tool call with empty function name");
      return;
    }

    // Trim whitespace from arguments
    std::string argsStr = state.current_arguments;
    argsStr.erase(0, argsStr.find_first_not_of(" \t\n\r"));
    argsStr.erase(argsStr.find_last_not_of(" \t\n\r") + 1);

    // Parse JSON arguments
    Json::Value argsJson;
    Json::CharReaderBuilder builder;
    std::string errs;
    std::istringstream jsonStream(argsStr);

    if (!Json::parseFromStream(builder, jsonStream, &argsJson, &errs)) {
      TT_LOG_WARN("[ToolCallParser] Failed to parse JSON arguments: {}", errs);
      return;
    }

    // Create tool call object in OpenAI format
    Json::Value toolCall;
    toolCall["id"] = "call_" + std::to_string(state.call_index);
    toolCall["type"] = "function";

    Json::Value function;
    function["name"] = state.current_function;

    // Convert arguments to string (OpenAI format expects string)
    Json::StreamWriterBuilder writerBuilder;
    writerBuilder["indentation"] = "";
    function["arguments"] = Json::writeString(writerBuilder, argsJson);

    toolCall["function"] = function;
    state.tool_calls.append(toolCall);

    TT_LOG_DEBUG("[ToolCallParser] Parsed tool call: name={}, args={}",
                 state.current_function, function["arguments"].asString());

    state.call_index++;
  }
};

}  // namespace

std::unique_ptr<IToolCallParser> createQwenToolCallParser();

std::unique_ptr<IToolCallParser> createToolCallParser(
    tt::config::ModelType modelType) {
  switch (modelType) {
    case tt::config::ModelType::DEEPSEEK_R1_0528:
      return std::make_unique<DeepSeekToolCallParser>();
    case tt::config::ModelType::LLAMA_3_1_8B_INSTRUCT:
      // TODO: Implement Llama tool call parser
      return std::make_unique<DeepSeekToolCallParser>();
    case tt::config::ModelType::QWEN_2_5_1_5B_INSTRUCT:
      return createQwenToolCallParser();
    default:
      return std::make_unique<DeepSeekToolCallParser>();
  }
}

}  // namespace tt::services
