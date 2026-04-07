// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

// Include fmt for template instantiation
#ifdef USE_EXTERNAL_FMT
#include <fmt/core.h>
#include <fmt/format.h>
#endif

namespace tt::utils {

/**
 * Zero-overhead, dependency-free high-performance logger.
 *
 * Features:
 * - True zero overhead for disabled log levels (compile-time elimination via
 * macros)
 * - Thread-safe operations
 * - File logging support
 * - Environment variable configuration
 * - No external dependencies - uses only standard C++
 * - Compatible with existing TT_LOG_* macros
 *
 * Environment Variables:
 * - TT_LOG_LEVEL: Set log level (trace, debug, info, warn, error, critical,
 * off)
 * - TT_LOG_FILE: Path to log file (optional, defaults to console only)
 *
 * Compile-time optimization:
 * Set TT_LOG_ACTIVE_LEVEL to eliminate logs below certain levels at compile
 * time. Example: -DTT_LOG_ACTIVE_LEVEL=3 to eliminate trace(0), debug(1),
 * info(2) logs.
 */
class ZeroOverheadLogger {
 public:
  enum Level {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CRITICAL = 5,
    OFF = 6
  };

  /**
   * Initialize logger with environment variable configuration.
   * This should be called once at application startup.
   */
  static void initialize();

  /**
   * Get the current log level
   */
  static Level getLevel() { return level; }

  /**
   * Log a message at the specified level
   */
  template <typename... Args>
  static void log(Level level, Args&&... args);

 private:
  static void logImpl(Level level, const char* fmtStr);
  template <typename... Args>
  static void logImpl(Level level, const char* fmtStr, Args&&... args);

  static Level parseLogLevel(const std::string& levelStr);
  static std::string levelToString(Level level);
  static std::shared_ptr<spdlog::logger> getLogger();

  static Level level;
  static std::shared_ptr<spdlog::logger> logger;
  static bool initialized;
};

// Template implementation
template <typename... Args>
void ZeroOverheadLogger::log(Level level, Args&&... args) {
  if (level < ZeroOverheadLogger::level) return;
  logImpl(level, std::forward<Args>(args)...);
}

// Template implementation for log_impl - must be in header for template
// instantiation
template <typename... Args>
void ZeroOverheadLogger::logImpl(Level level, const char* fmtStr,
                                 Args&&... args) {
  auto logger = getLogger();

// Use runtime format string - spdlog handles fmt internally
#ifdef USE_EXTERNAL_FMT
  auto runtimeFmt = fmt::runtime(fmtStr);
#else
  auto runtimeFmt = spdlog::fmt_lib::runtime(fmtStr);
#endif

  switch (level) {
    case TRACE:
      logger->trace(runtimeFmt, std::forward<Args>(args)...);
      break;
    case DEBUG:
      logger->debug(runtimeFmt, std::forward<Args>(args)...);
      break;
    case INFO:
      logger->info(runtimeFmt, std::forward<Args>(args)...);
      break;
    case WARN:
      logger->warn(runtimeFmt, std::forward<Args>(args)...);
      break;
    case ERROR:
      logger->error(runtimeFmt, std::forward<Args>(args)...);
      break;
    case CRITICAL:
      logger->critical(runtimeFmt, std::forward<Args>(args)...);
      break;
    case OFF:
      break;
  }
}

// Compile-time level checking for zero overhead
#ifndef TT_LOG_ACTIVE_LEVEL
#define TT_LOG_ACTIVE_LEVEL 0  // Include all levels by default
#endif

// Zero-overhead macros - completely eliminated if below active level
#if TT_LOG_ACTIVE_LEVEL <= 0
#define TT_LOG_TRACE_ZERO(...)                                             \
  tt::utils::ZeroOverheadLogger::log(tt::utils::ZeroOverheadLogger::TRACE, \
                                     __VA_ARGS__)
#else
#define TT_LOG_TRACE_ZERO(...) \
  do {                         \
  } while (0)
#endif

#if TT_LOG_ACTIVE_LEVEL <= 1
#define TT_LOG_DEBUG_ZERO(...)                                             \
  tt::utils::ZeroOverheadLogger::log(tt::utils::ZeroOverheadLogger::DEBUG, \
                                     __VA_ARGS__)
#else
#define TT_LOG_DEBUG_ZERO(...) \
  do {                         \
  } while (0)
#endif

#if TT_LOG_ACTIVE_LEVEL <= 2
#define TT_LOG_INFO_ZERO(...)                                             \
  tt::utils::ZeroOverheadLogger::log(tt::utils::ZeroOverheadLogger::INFO, \
                                     __VA_ARGS__)
#else
#define TT_LOG_INFO_ZERO(...) \
  do {                        \
  } while (0)
#endif

#if TT_LOG_ACTIVE_LEVEL <= 3
#define TT_LOG_WARN_ZERO(...)                                             \
  tt::utils::ZeroOverheadLogger::log(tt::utils::ZeroOverheadLogger::WARN, \
                                     __VA_ARGS__)
#else
#define TT_LOG_WARN_ZERO(...) \
  do {                        \
  } while (0)
#endif

#if TT_LOG_ACTIVE_LEVEL <= 4
#define TT_LOG_ERROR_ZERO(...)                                             \
  tt::utils::ZeroOverheadLogger::log(tt::utils::ZeroOverheadLogger::ERROR, \
                                     __VA_ARGS__)
#else
#define TT_LOG_ERROR_ZERO(...) \
  do {                         \
  } while (0)
#endif

#if TT_LOG_ACTIVE_LEVEL <= 5
#define TT_LOG_CRITICAL_ZERO(...)                                             \
  tt::utils::ZeroOverheadLogger::log(tt::utils::ZeroOverheadLogger::CRITICAL, \
                                     __VA_ARGS__)
#else
#define TT_LOG_CRITICAL_ZERO(...) \
  do {                            \
  } while (0)
#endif

// Main logging macros - always defined
#define TT_LOG_TRACE(...) TT_LOG_TRACE_ZERO(__VA_ARGS__)
#define TT_LOG_DEBUG(...) TT_LOG_DEBUG_ZERO(__VA_ARGS__)
#define TT_LOG_INFO(...) TT_LOG_INFO_ZERO(__VA_ARGS__)
#define TT_LOG_WARN(...) TT_LOG_WARN_ZERO(__VA_ARGS__)
#define TT_LOG_ERROR(...) TT_LOG_ERROR_ZERO(__VA_ARGS__)
#define TT_LOG_CRITICAL(...) TT_LOG_CRITICAL_ZERO(__VA_ARGS__)

}  // namespace tt::utils
