#pragma once

#include <chrono>
#include <iostream>

#ifdef LLM_ENGINE_DEBUG

namespace llm_engine::detail {

inline double elapsed_ms() {
  static const auto start = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - start).count();
}

}  // namespace llm_engine::detail

#  define LLM_ENGINE_LOG(component)                                 \
    std::cerr << "[" << llm_engine::detail::elapsed_ms() << " ms " \
              << "llm_engine:" << (component) << "] "
#else
#  define LLM_ENGINE_LOG(component) \
    if (false) std::cerr
#endif
