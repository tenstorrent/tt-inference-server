#pragma once

#include "llm_engine/config.hpp"

namespace llm_engine {

/** Creates mesh device, H2D/D2H sockets, fills config and returns opaque context. Caller owns context and must call destroy. */
void* create_ttmetal_decode_context_and_config(Config* config);
/** Closes device and frees context. */
void destroy_ttmetal_decode_context(void* ctx);

}  // namespace llm_engine
