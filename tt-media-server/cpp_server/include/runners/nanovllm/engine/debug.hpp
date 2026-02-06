#pragma once

#include <iostream>

#ifdef NANOVLLM_USE_EXTERNAL_LOG
#  include "logging/stream_log.hpp"
#  define NANOVLLM_LOG(component) ::tt::logging::StreamLog("DEBUG", "nanovllm:" component)
#elif defined(NANOVLLM_DEBUG)
#  define NANOVLLM_LOG(component) \
    std::cerr << "[nanovllm:" << (component) << "] "
#else
#  define NANOVLLM_LOG(component) \
    if (false) std::cerr
#endif
