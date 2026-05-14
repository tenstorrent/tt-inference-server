// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <json/reader.h>
#include <json/value.h>
#include <json/writer.h>

#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "services/tool_call_parser.hpp"
#include "utils/logger.hpp"
#include "utils/tool_call_id_generator.hpp"

namespace tt::services {

namespace {

constexpr const char* kOpenTag = "<tool_call>";
constexpr const char* kCloseTag = "</tool_call>";
constexpr size_t kOpenTagLen = 11;
constexpr size_t kCloseTagLen = 12;

enum class State {
  REGULAR,       // Watching for <tool_call>
  IN_TOOL_CALL,  // Between <tool_call> and </tool_call>
};

struct TaskState {
  State state = State::REGULAR;
  std::string buffer;       // text accumulator
  std::string innerBuffer;  // JSON between <tool_call> and </tool_call>
  Json::Value toolCalls = Json::Value(Json::arrayValue);
  int callIndex = 0;
};

Json::Value buildBundledDelta(int index, const std::string& callId,
                              const std::string& functionName,
                              const std::string& argumentsJson) {
  Json::Value toolCallsArray(Json::arrayValue);
  Json::Value entry;
  entry["index"] = index;
  entry["id"] = callId;
  entry["type"] = "function";
  entry["function"]["name"] = functionName;
  entry["function"]["arguments"] = argumentsJson;
  toolCallsArray.append(entry);
  return toolCallsArray;
}

/**
 * Qwen tool-call parser.
 *
 * Qwen emits tool calls as:
 *   <tool_call>
 *   {"name": "func_name", "arguments": {...}}
 *   </tool_call>
 *
 * Strategy: accumulate decoded text, switch to IN_TOOL_CALL on the opening
 * tag (suppressing model output while inside), and on the closing tag parse
 * the buffered JSON and emit a single bundled tool_call delta carrying id,
 * function name, and the full arguments JSON. The NonStreamResponseWriter
 * accumulator handles this correctly; streaming clients receive the tool call
 * as one chunk rather than incremental argument bytes — adequate for a
 * non-batching CPU runner.
 */
class QwenToolCallParser : public IToolCallParser {
 public:
  void initializeTask(uint32_t taskId) override {
    std::lock_guard<std::mutex> lock(mutex_);
    taskStates_[taskId] = TaskState{};
    TT_LOG_DEBUG("[QwenToolCallParser] Initialized task {}", taskId);
  }

  std::optional<ToolCallTokenResult> processToken(
      uint32_t taskId, [[maybe_unused]] int64_t tokenId,
      const std::string& decodedText) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = taskStates_.find(taskId);
    if (it == taskStates_.end()) return std::nullopt;
    TaskState& s = it->second;

    s.buffer += decodedText;

    while (true) {
      if (s.state == State::REGULAR) {
        size_t pos = s.buffer.find(kOpenTag);
        if (pos == std::string::npos) {
          // No opening tag yet. Trim safely consumable prefix so the buffer
          // can't grow unbounded; keep enough tail to match a future tag.
          if (s.buffer.size() > kOpenTagLen) {
            s.buffer = s.buffer.substr(s.buffer.size() - kOpenTagLen);
          }
          return std::nullopt;
        }
        // Consume up to and including the opening tag and switch states.
        s.buffer.erase(0, pos + kOpenTagLen);
        s.innerBuffer.clear();
        s.state = State::IN_TOOL_CALL;
        continue;
      }

      // IN_TOOL_CALL: search for closing tag.
      size_t pos = s.buffer.find(kCloseTag);
      if (pos == std::string::npos) {
        // Move all but a tail (the length of the close tag minus 1) into
        // innerBuffer; keep the tail so we can finish matching a split tag.
        if (s.buffer.size() > kCloseTagLen - 1) {
          size_t take = s.buffer.size() - (kCloseTagLen - 1);
          s.innerBuffer.append(s.buffer, 0, take);
          s.buffer.erase(0, take);
        }
        return std::nullopt;
      }

      s.innerBuffer.append(s.buffer, 0, pos);
      s.buffer.erase(0, pos + kCloseTagLen);
      s.state = State::REGULAR;

      auto delta = finalizeOneToolCall(s);
      // Return only the first complete tool call delta from this token. Any
      // additional tool-call text still in the buffer will be picked up on
      // the next token (rare for Qwen which usually emits one block).
      if (delta.has_value()) return delta;
    }
  }

  std::optional<Json::Value> finalizeTask(uint32_t taskId) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = taskStates_.find(taskId);
    if (it == taskStates_.end()) return std::nullopt;
    std::optional<Json::Value> result;
    if (!it->second.toolCalls.empty()) result = it->second.toolCalls;
    taskStates_.erase(it);
    return result;
  }

  bool isInToolCall(uint32_t taskId) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = taskStates_.find(taskId);
    if (it == taskStates_.end()) return false;
    return it->second.state == State::IN_TOOL_CALL;
  }

  size_t activeTaskCount() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return taskStates_.size();
  }

 private:
  std::optional<ToolCallTokenResult> finalizeOneToolCall(TaskState& s) {
    Json::CharReaderBuilder rbuilder;
    Json::Value parsed;
    std::string errs;
    std::istringstream iss(s.innerBuffer);
    if (!Json::parseFromStream(rbuilder, iss, &parsed, &errs)) {
      TT_LOG_WARN("[QwenToolCallParser] Failed to parse tool-call JSON: {}",
                  errs);
      s.innerBuffer.clear();
      return std::nullopt;
    }

    std::string functionName;
    if (parsed.isMember("name") && parsed["name"].isString()) {
      functionName = parsed["name"].asString();
    }
    if (functionName.empty()) {
      TT_LOG_WARN("[QwenToolCallParser] Tool-call JSON missing 'name'");
      s.innerBuffer.clear();
      return std::nullopt;
    }

    // Serialize arguments as a JSON string (OpenAI format expects a string).
    Json::StreamWriterBuilder wbuilder;
    wbuilder["indentation"] = "";
    std::string argsStr;
    if (parsed.isMember("arguments")) {
      const auto& args = parsed["arguments"];
      argsStr =
          args.isString() ? args.asString() : Json::writeString(wbuilder, args);
    } else {
      argsStr = "{}";
    }

    const std::string callId = tt::utils::ToolCallIDGenerator::generate();

    Json::Value toolCall;
    toolCall["id"] = callId;
    toolCall["type"] = "function";
    toolCall["function"]["name"] = functionName;
    toolCall["function"]["arguments"] = argsStr;
    s.toolCalls.append(toolCall);

    const int idx = s.callIndex++;
    s.innerBuffer.clear();

    TT_LOG_DEBUG("[QwenToolCallParser] Parsed tool call name={} id={} args={}",
                 functionName, callId, argsStr);

    return ToolCallTokenResult{
        ToolCallDeltaType::TOOL_CALL_START,
        idx,
        argsStr,
        functionName,
        callId,
        buildBundledDelta(idx, callId, functionName, argsStr)};
  }

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, TaskState> taskStates_;
};

}  // namespace

std::unique_ptr<IToolCallParser> createQwenToolCallParser() {
  return std::make_unique<QwenToolCallParser>();
}

}  // namespace tt::services
