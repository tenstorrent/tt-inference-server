// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/tool_call/deepseek_tool_call_parser.hpp"
#include "services/tool_call/mock_tool_call_parser.hpp"
#include "services/tool_call/tool_call_parser.hpp"

namespace tt::services::tool_call {

std::unique_ptr<ToolCallParser> createToolCallParser(
    tt::config::ModelRunnerType runnerType) {
  switch (runnerType) {
    case tt::config::ModelRunnerType::MOCK:
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return std::make_unique<MockToolCallParser>();

    case tt::config::ModelRunnerType::PIPELINE:
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
    case tt::config::ModelRunnerType::PREFILL:
      return std::make_unique<DeepSeekToolCallParser>();

    case tt::config::ModelRunnerType::LLAMA:
      // Llama tool calling not yet implemented
      return std::make_unique<MockToolCallParser>();

    default:
      // Default to mock parser for unknown runner types
      return std::make_unique<MockToolCallParser>();
  }
}

}  // namespace tt::services::tool_call
