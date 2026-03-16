#pragma once

#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>

#ifdef LLM_ENGINE_DEBUG

namespace llm_engine::detail {

inline double elapsed_ms() {
  static const auto start = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - start).count();
}

inline std::mutex& log_mutex() {
  static std::mutex m;
  return m;
}

class LogBuf {
 public:
  explicit LogBuf(const char* component) : buf_() {
    buf_ << "[" << elapsed_ms() << " ms llm_engine:" << component << "] ";
  }
  ~LogBuf() {
    std::lock_guard<std::mutex> lock(log_mutex());
    std::cerr << buf_.str();
  }
  template <typename T>
  LogBuf& operator<<(const T& v) {
    buf_ << v;
    return *this;
  }
  LogBuf& operator<<(std::ostream& (*manip)(std::ostream&)) {
    buf_ << manip;
    return *this;
  }

 private:
  std::ostringstream buf_;
};

}  // namespace llm_engine::detail

#define LLM_ENGINE_LOG(component) llm_engine::detail::LogBuf(component)
#else
#define LLM_ENGINE_LOG(component) \
  if (false) std::cerr
#endif
