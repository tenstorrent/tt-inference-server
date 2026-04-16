// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config/types.hpp"
#include "domain/tool.hpp"

namespace tt::services::tool_call {

/**
 * Base class for tool call parsers.
 * Different models use different tool calling formats, so subclasses
 * implement model-specific parsing logic.
 */
class ToolCallParser {
 public:
  virtual ~ToolCallParser() = default;

  /**
   * Parse complete text to extract tool calls.
   * Used for non-streaming requests.
   *
   * @param text The complete model-generated text
   * @return Optional vector of ToolCall objects, or nullopt if no tool calls
   * found
   */
  virtual std::optional<std::vector<domain::ToolCall>> parseComplete(
      const std::string& text) const = 0;

  /**
   * Strip tool call markers from text, keeping any text before/after.
   * Each parser implementation handles its own marker format.
   *
   * @param text The text containing tool call markers
   * @return Text with tool call markers removed
   */
  virtual std::string stripMarkers(const std::string& text) const = 0;

 protected:
  ToolCallParser() = default;
};

/**
 * Factory function to create the appropriate parser based on runner type.
 *
 * @param runnerType The model runner type
 * @return Unique pointer to the appropriate ToolCallParser implementation
 */
std::unique_ptr<ToolCallParser> createToolCallParser(
    tt::config::ModelRunnerType runnerType);

}  // namespace tt::services::tool_call
