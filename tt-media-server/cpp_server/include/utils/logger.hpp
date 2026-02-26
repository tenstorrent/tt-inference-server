// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <cstdlib>
#include <algorithm>

namespace tt::utils {

/**
 * Simple singleton logger using only standard C++ libraries.
 *
 * Features:
 * - Singleton pattern for global access
 * - Environment variable configuration
 * - File logging support with rotation
 * - Thread-safe operations
 * - Simple format string replacement using {}
 *
 * Environment Variables:
 * - TT_LOG_LEVEL: Set log level (trace, debug, info, warn, error, critical, off)
 * - TT_LOG_FILE: Path to log file (optional, defaults to console only)
 * - TT_LOG_MAX_SIZE: Maximum log file size in bytes (default: 10MB)
 * - TT_LOG_MAX_FILES: Maximum number of rotating log files (default: 3)
 */
class Logger {
public:
    enum class Level {
        TRACE = 0,
        DEBUG = 1,
        INFO = 2,
        WARN = 3,
        ERROR = 4,
        CRITICAL = 5,
        OFF = 6
    };

    /**
     * Get the singleton logger instance.
     * Thread-safe lazy initialization.
     */
    static std::shared_ptr<Logger> get_logger();

    /**
     * Simple log methods
     */
    void log_trace(const std::string& message) { log(Level::TRACE, message); }
    void log_debug(const std::string& message) { log(Level::DEBUG, message); }
    void log_info(const std::string& message) { log(Level::INFO, message); }
    void log_warning(const std::string& message) { log(Level::WARN, message); }
    void log_error(const std::string& message) { log(Level::ERROR, message); }
    void log_critical(const std::string& message) { log(Level::CRITICAL, message); }

    /**
     * Formatted log methods using simple {} replacement
     */
    template<typename... Args>
    void log_trace(const std::string& format, Args&&... args) {
        if (level_ <= Level::TRACE) {
            log(Level::TRACE, simple_format(format, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void log_debug(const std::string& format, Args&&... args) {
        if (level_ <= Level::DEBUG) {
            log(Level::DEBUG, simple_format(format, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void log_info(const std::string& format, Args&&... args) {
        if (level_ <= Level::INFO) {
            log(Level::INFO, simple_format(format, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void log_warning(const std::string& format, Args&&... args) {
        if (level_ <= Level::WARN) {
            log(Level::WARN, simple_format(format, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void log_error(const std::string& format, Args&&... args) {
        if (level_ <= Level::ERROR) {
            log(Level::ERROR, simple_format(format, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void log_critical(const std::string& format, Args&&... args) {
        if (level_ <= Level::CRITICAL) {
            log(Level::CRITICAL, simple_format(format, std::forward<Args>(args)...));
        }
    }

    /**
     * Set log level
     */
    void set_level(Level level) { level_ = level; }

    /**
     * Flush all output streams
     */
    void flush();

private:
    Logger();

    // Custom deleter for shared_ptr (allows private destructor)
    struct Deleter {
        void operator()(Logger* logger) {
            delete logger;
        }
    };

    friend struct Deleter;
    ~Logger() = default;

    // Delete copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void initialize();
    Level parse_log_level(const std::string& level_str);
    std::string level_to_string(Level level);
    std::string get_timestamp();
    void log(Level level, const std::string& message);
    void rotate_file_if_needed();

    template<typename T>
    std::string to_string_helper(T&& value) {
        if constexpr (std::is_convertible_v<T, std::string>) {
            return std::string(std::forward<T>(value));
        } else {
            std::ostringstream oss;
            oss << std::forward<T>(value);
            return oss.str();
        }
    }

    template<typename... Args>
    std::string simple_format(const std::string& format, Args&&... args) {
        if constexpr (sizeof...(args) == 0) {
            return format;
        } else {
            std::string result = format;
            size_t pos = 0;

            auto replace_next = [&](auto&& arg) {
                size_t found = result.find("{}", pos);
                if (found != std::string::npos) {
                    std::string arg_str = to_string_helper(std::forward<decltype(arg)>(arg));
                    result.replace(found, 2, arg_str);
                    pos = found + arg_str.length();
                }
            };

            (replace_next(args), ...);
            return result;
        }
    }

    Level level_;
    std::string log_file_;
    size_t max_file_size_;
    size_t max_files_;
    std::ofstream file_stream_;
    mutable std::mutex mutex_;

    static std::shared_ptr<Logger> instance_;
    static std::mutex static_mutex_;
};

} // namespace tt::utils

// Convenience macros for easy logging
#define TT_LOG_TRACE(...) tt::utils::Logger::get_logger()->log_trace(__VA_ARGS__)
#define TT_LOG_DEBUG(...) tt::utils::Logger::get_logger()->log_debug(__VA_ARGS__)
#define TT_LOG_INFO(...)  tt::utils::Logger::get_logger()->log_info(__VA_ARGS__)
#define TT_LOG_WARN(...)  tt::utils::Logger::get_logger()->log_warning(__VA_ARGS__)
#define TT_LOG_ERROR(...) tt::utils::Logger::get_logger()->log_error(__VA_ARGS__)
#define TT_LOG_CRITICAL(...) tt::utils::Logger::get_logger()->log_critical(__VA_ARGS__)
