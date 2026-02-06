#pragma once

#include <iostream>

#ifdef NANOVLLM_USE_EXTERNAL_LOG
#  include "logging/nanovllm_log.hpp"
#elif defined(NANOVLLM_DEBUG)
#  define NANOVLLM_LOG(component) \
    std::cerr << "[nanovllm:" << (component) << "] "
#else
#  define NANOVLLM_LOG(component) \
    if (false) std::cerr
#endif
