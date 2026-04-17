// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/tool_call_parser.hpp"

#include <json/reader.h>
#include <json/value.h>

#include <regex>

#include "config/types.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {

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
  std::optional<Json::Value> parseComplete(
      const std::string& text) const override {
    // Look for tool call markers
    if (text.find("<｜tool▁calls▁begin｜>") == std::string::npos) {
      return std::nullopt;
    }

    Json::Value toolCallsArray(Json::arrayValue);
    int callIndex = 0;

    // Pattern to match: function<｜tool▁sep｜>function_name\n```json\n{...}\n```
    // Using a simpler approach: find the markers and extract content between
    // them
    size_t pos = 0;
    while (true) {
      // Find next tool call
      size_t callBegin = text.find("<｜tool▁call▁begin｜>", pos);
      if (callBegin == std::string::npos) break;

      size_t callEnd = text.find("<｜tool▁call▁end｜>", callBegin);
      if (callEnd == std::string::npos) break;

      // Extract the content between markers
      std::string callContent =
          text.substr(callBegin + 20, callEnd - (callBegin + 20));

      // Extract function name (after "function<｜tool▁sep｜>")
      size_t sepPos = callContent.find("<｜tool▁sep｜>");
      if (sepPos == std::string::npos) {
        pos = callEnd + 1;
        continue;
      }

      // Function name is between separator and newline/backticks
      size_t nameStart = sepPos + 13;  // Length of "<｜tool▁sep｜>"
      size_t nameEnd = callContent.find("\n", nameStart);
      if (nameEnd == std::string::npos) {
        nameEnd = callContent.find("```", nameStart);
      }
      if (nameEnd == std::string::npos) {
        pos = callEnd + 1;
        continue;
      }

      std::string functionName =
          callContent.substr(nameStart, nameEnd - nameStart);
      // Trim whitespace
      functionName.erase(0, functionName.find_first_not_of(" \t\n\r"));
      functionName.erase(functionName.find_last_not_of(" \t\n\r") + 1);

      // Extract JSON arguments (between ```json and ```)
      size_t jsonStart = callContent.find("```json");
      if (jsonStart == std::string::npos) {
        // Try without the "json" specifier
        jsonStart = callContent.find("```");
      } else {
        jsonStart += 7;  // Skip "```json"
      }

      if (jsonStart != std::string::npos) {
        jsonStart = callContent.find_first_not_of(" \t\n\r", jsonStart);
        size_t jsonEnd = callContent.find("```", jsonStart);

        if (jsonEnd != std::string::npos) {
          std::string jsonStr =
              callContent.substr(jsonStart, jsonEnd - jsonStart);
          // Trim whitespace
          jsonStr.erase(0, jsonStr.find_first_not_of(" \t\n\r"));
          jsonStr.erase(jsonStr.find_last_not_of(" \t\n\r") + 1);

          // Parse JSON arguments
          Json::Value argsJson;
          Json::CharReaderBuilder builder;
          std::string errs;
          std::istringstream jsonStream(jsonStr);

          if (Json::parseFromStream(builder, jsonStream, &argsJson, &errs)) {
            // Create tool call object
            Json::Value toolCall;
            toolCall["id"] = "call_" + std::to_string(callIndex);
            toolCall["type"] = "function";

            Json::Value function;
            function["name"] = functionName;
            // Convert arguments to string (OpenAI format expects string)
            Json::StreamWriterBuilder writerBuilder;
            writerBuilder["indentation"] = "";
            function["arguments"] =
                Json::writeString(writerBuilder, argsJson);

            toolCall["function"] = function;
            toolCallsArray.append(toolCall);

            TT_LOG_DEBUG(
                "[DeepSeekToolCallParser] Parsed tool call: name={}, "
                "args={}",
                functionName, function["arguments"].asString());

            callIndex++;
          } else {
            TT_LOG_WARN(
                "[DeepSeekToolCallParser] Failed to parse JSON arguments: {}",
                errs);
          }
        }
      }

      pos = callEnd + 1;
    }

    if (toolCallsArray.empty()) {
      return std::nullopt;
    }

    return toolCallsArray;
  }

  std::string stripMarkers(const std::string& text) const override {
    // Remove tool call markers and content
    std::string result = text;

    // Remove everything between <｜tool▁calls▁begin｜> and
    // <｜tool▁calls▁end｜>
    size_t startPos = result.find("<｜tool▁calls▁begin｜>");
    if (startPos != std::string::npos) {
      size_t endPos = result.find("<｜tool▁calls▁end｜>", startPos);
      if (endPos != std::string::npos) {
        result.erase(startPos, endPos - startPos + 19);
      }
    }

    // Trim whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);

    return result;
  }
};

}  // namespace

std::unique_ptr<IToolCallParser> createToolCallParser(
    tt::config::ModelType modelType) {
  switch (modelType) {
    case tt::config::ModelType::DEEPSEEK_R1_0528:
      return std::make_unique<DeepSeekToolCallParser>();
    case tt::config::ModelType::LLAMA_3_1_8B_INSTRUCT:
      // TODO: Implement Llama tool call parser
      return std::make_unique<DeepSeekToolCallParser>();
    default:
      return std::make_unique<DeepSeekToolCallParser>();
  }
}

}  // namespace tt::services
