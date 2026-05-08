#pragma once

#include "domain/llm/llm_request.hpp"
#include "domain/llm/sampling_params.hpp"

namespace tt::utils::mapper {

using namespace tt::domain::llm;

SamplingParams mapSamplingParams(const LLMRequest&);
}  // namespace tt::utils::mapper