#include "llm_engine/engine/device_context_ttmetal.hpp"

namespace llm_engine {

void* create_ttmetal_decode_context_and_config(Config* config) {
  (void)config;
  return nullptr;
}

void destroy_ttmetal_decode_context(void* ctx) {
  (void)ctx;
}

}  // namespace llm_engine
