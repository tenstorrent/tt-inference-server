#pragma once

#include <cstdint>
#include <vector>

namespace llm_engine {

int64_t hash_token_ids(const std::vector<int64_t>& token_ids, int64_t prefix);

}  // namespace llm_engine
