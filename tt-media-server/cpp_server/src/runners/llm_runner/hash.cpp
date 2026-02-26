#include "runners/llm_runner/hash.hpp"
#include <cstring>

namespace llm_engine {

static constexpr uint64_t kFnvPrime = 1099511628211ULL;
static constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;

int64_t hash_token_ids(const std::vector<int64_t>& token_ids, int64_t prefix) {
  uint64_t h = kFnvOffset;
  if (prefix != -1) {
    uint64_t p = static_cast<uint64_t>(prefix);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&p);
    for (size_t i = 0; i < sizeof(p); ++i) {
      h ^= bytes[i];
      h *= kFnvPrime;
    }
  }
  const uint8_t* data = reinterpret_cast<const uint8_t*>(token_ids.data());
  size_t len = token_ids.size() * sizeof(int64_t);
  for (size_t i = 0; i < len; ++i) {
    h ^= data[i];
    h *= kFnvPrime;
  }
  return static_cast<int64_t>(h);
}

}  // namespace llm_engine
