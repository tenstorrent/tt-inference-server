#pragma once

#include "domain/llm_request.hpp"
#include "runners/llm_runner/sampling_params.hpp"

namespace tt::utils::mapper {

tt::runners::llm_engine::SamplingParams mapSamplingParams(
    const domain::LLMRequest&);
}