#include "utils/mapper.hpp"

namespace tt::utils::mapper {

namespace {
llm_engine::ResponseFormatType mapResponseFormatType(
    tt::domain::ResponseFormatType t) {
  switch (t) {
    case tt::domain::ResponseFormatType::JSON_OBJECT:
      return llm_engine::ResponseFormatType::JSON_OBJECT;
    case tt::domain::ResponseFormatType::JSON_SCHEMA:
      return llm_engine::ResponseFormatType::JSON_SCHEMA;
    default:
      return llm_engine::ResponseFormatType::TEXT;
  }
}
}  // namespace

llm_engine::SamplingParams mapSamplingParams(
    const tt::domain::LLMRequest& request) {
  llm_engine::SamplingParams params;
  params.temperature = request.temperature.value_or(1.0f);
  params.max_tokens = request.max_tokens;
  params.ignore_eos = request.ignore_eos;
  params.top_p = request.top_p;
  params.presence_penalty = request.presence_penalty;
  params.frequency_penalty = request.frequency_penalty;
  params.seed = request.seed;
  params.use_beam_search = request.use_beam_search;
  params.top_k = request.top_k;
  params.min_p = request.min_p;
  params.repetition_penalty = request.repetition_penalty;
  params.length_penalty = request.length_penalty;
  params.stop_token_ids = request.stop_token_ids;
  params.include_stop_str_in_output = request.include_stop_str_in_output;
  params.min_tokens = request.min_tokens;
  params.skip_special_tokens = request.skip_special_tokens;
  params.spaces_between_special_tokens = request.spaces_between_special_tokens;
  params.allowed_token_ids = request.allowed_token_ids;
  params.prompt_logprobs = request.prompt_logprobs;
  params.truncate_prompt_tokens = request.truncate_prompt_tokens;
  params.fast_mode = request.fast_mode;

  if (request.response_format.has_value()) {
    params.response_format_type =
        mapResponseFormatType(request.response_format->type);
    params.json_schema_str = request.response_format->json_schema_str;
  }

  return params;
}

}  // namespace tt::utils::mapper
