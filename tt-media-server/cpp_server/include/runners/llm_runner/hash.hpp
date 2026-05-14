#pragma once

#include <cstdint>
#include <vector>

namespace tt::runners::llm_engine {

int64_t hashTokenIds(const std::vector<int64_t>& tokenIds, int64_t prefix);

}  // namespace tt::runners::llm_engine
