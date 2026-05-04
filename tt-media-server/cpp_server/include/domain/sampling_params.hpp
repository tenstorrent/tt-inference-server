#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config/types.hpp"
#include "domain/tool_calls/tool.hpp"
#include "domain/tool_calls/tool_choice.hpp"

namespace tt::domain {

using tt::config::ResponseFormatType;

/**
 * Sampling parameters aligned with OpenAI-compatible completion request.
 * Mirrors sampling-related fields from tt::domain::LLMRequest.
 */
struct SamplingParams {
  float temperature = 0.0f;
  std::optional<int> max_tokens;
  bool ignore_eos = false;

  std::optional<float> top_p;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  std::optional<int> seed;

  bool use_beam_search = false;
  std::optional<int> top_k;
  std::optional<float> min_p;
  std::optional<float> repetition_penalty;
  float length_penalty = 1.0f;

  std::vector<int> stop_token_ids;
  bool include_stop_str_in_output = false;
  int min_tokens = 0;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
  std::optional<std::vector<int>> allowed_token_ids;
  std::optional<std::vector<int32_t>> token_bitmask;
  int bitmask_vocab_size = 0;
  std::optional<int> prompt_logprobs;
  std::optional<int> truncate_prompt_tokens;
  bool fast_mode = false;

  ResponseFormatType response_format_type = ResponseFormatType::TEXT;
  std::optional<std::string> json_schema_str;

  std::optional<std::vector<tool_calls::Tool>> tools;
  std::optional<tool_calls::ToolChoice> tool_choice;

  bool hasGuidedDecoding() const {
    if (response_format_type != ResponseFormatType::TEXT) {
      return true;
    }
    if (tool_choice.has_value() && tool_choice->type == "function" &&
        tool_choice->function.has_value() && tools.has_value()) {
      return true;
    }
    return false;
  }

  void serialize(std::ostream& os) const;
  static std::unique_ptr<SamplingParams> deserialize(std::istream& is);
};

}  // namespace tt::domain
