#pragma once

#include <iostream>

#ifdef LLM_ENGINE_USE_EXTERNAL_LOG
#  include "logging/stream_log.hpp"
#  define LLM_ENGINE_LOG(component) ::tt::logging::StreamLog("DEBUG", "llm_engine:" component)
#elif defined(LLM_ENGINE_DEBUG)
#  define LLM_ENGINE_LOG(component) \
    std::cerr << "[llm_engine:" << (component) << "] "
#else
#  define LLM_ENGINE_LOG(component) \
    if (false) std::cerr
#endif
