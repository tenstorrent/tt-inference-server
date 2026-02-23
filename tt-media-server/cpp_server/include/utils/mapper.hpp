#include "domain/completion_request.hpp"
#include "runners/llm_runner/sampling_params.hpp"

namespace tt::utils::mapper {

llm_engine::SamplingParams map_sampling_params(const domain::CompletionRequest&);

}