// Debug-mode instrumentation helper (session 680f96). Temporary; remove after
// profiling.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>

namespace tt::debug {

inline std::mutex& dbg_mutex() {
  static std::mutex m;
  return m;
}

inline FILE*& dbg_file() {
  static FILE* f = std::fopen(
      "/localdev/knovokmet/tt-inference-server/tt-media-server/cpp_server/"
      ".cursor/debug-680f96.log",
      "ae");
  return f;
}

inline uint64_t dbg_now_us() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline void dbg_log(const char* location, const char* message,
                    const std::string& data_json) {
  std::lock_guard<std::mutex> lock(dbg_mutex());
  FILE* f = dbg_file();
  if (!f) return;
  std::fprintf(f,
               "{\"sessionId\":\"680f96\",\"location\":\"%s\","
               "\"message\":\"%s\",\"data\":%s,\"t_us\":%llu}\n",
               location, message, data_json.c_str(),
               static_cast<unsigned long long>(dbg_now_us()));
  std::fflush(f);
}

}  // namespace tt::debug

#define DBG_LOG(location, message, json_body)                    \
  do {                                                           \
    std::ostringstream _dbg_oss;                                 \
    _dbg_oss << json_body;                                       \
    ::tt::debug::dbg_log((location), (message), _dbg_oss.str()); \
  } while (0)
