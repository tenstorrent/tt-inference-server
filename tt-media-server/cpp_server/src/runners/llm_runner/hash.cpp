#include "runners/llm_runner/hash.hpp"

#include <cstring>

namespace tt::runners::llm_engine {

static constexpr uint64_t K_FNV_PRIME = 1099511628211ULL;
static constexpr uint64_t K_FNV_OFFSET = 0xcbf29ce484222325ULL;

int64_t hashTokenIds(const std::vector<int64_t>& tokenIds, int64_t prefix) {
  uint64_t h = K_FNV_OFFSET;
  if (prefix != -1) {
    uint64_t p = static_cast<uint64_t>(prefix);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&p);
    for (size_t i = 0; i < sizeof(p); ++i) {
      h ^= bytes[i];
      h *= K_FNV_PRIME;
    }
  }
  const uint8_t* data = reinterpret_cast<const uint8_t*>(tokenIds.data());
  size_t len = tokenIds.size() * sizeof(int64_t);
  for (size_t i = 0; i < len; ++i) {
    h ^= data[i];
    h *= K_FNV_PRIME;
  }
  return static_cast<int64_t>(h);
}

}  // namespace tt::runners::llm_engine
