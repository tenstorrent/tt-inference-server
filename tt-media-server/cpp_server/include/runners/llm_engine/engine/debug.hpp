#pragma once

#include <iostream>

#ifdef LLM_ENGINE_DEBUG
#  define LLM_ENGINE_LOG(component) \
    std::cerr << "[llm_engine:" << (component) << "] "
#else
#  define LLM_ENGINE_LOG(component) \
    if (false) std::cerr
#endif
