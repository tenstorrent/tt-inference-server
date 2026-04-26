#pragma once

#include "domain/llm_request.hpp"
#include "domain/sampling_params.hpp"

namespace tt::utils::mapper {

tt::domain::SamplingParams mapSamplingParams(const domain::LLMRequest&);
}