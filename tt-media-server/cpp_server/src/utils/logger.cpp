// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "utils/logger.hpp"
#include <filesystem>
#include <iostream>
#include <cstring>
#include <cstdlib>

namespace tt::utils {

std::shared_ptr<Logger> Logger::instance_ = nullptr;
std::mutex Logger::static_mutex_;

std::shared_ptr<Logger> Logger::get_logger() {
    std::lock_guard<std::mutex> lock(static_mutex_);
    if (instance_ == nullptr) {
        instance_ = std::shared_ptr<Logger>(new Logger(), Deleter{});
    }
    return instance_;
}

Logger::Logger() : level_(Level::INFO), max_file_size_(10 * 1024 * 1024), max_files_(3) {
    initialize();
}

void Logger::initialize() {
    try {
        // Set log level from environment variable
        const char* log_level_env = std::getenv("TT_LOG_LEVEL");
        if (log_level_env) {
            level_ = parse_log_level(log_level_env);
        }

        // Check for file logging configuration
        const char* log_file_env = std::getenv("TT_LOG_FILE");
        if (log_file_env && strlen(log_file_env) > 0) {
            log_file_ = log_file_env;

            // Get file rotation settings
            const char* max_size_env = std::getenv("TT_LOG_MAX_SIZE");
            if (max_size_env) {
                try {
                    size_t size_bytes = std::stoul(max_size_env);
                    max_file_size_ = size_bytes;
                } catch (const std::exception&) {
                    std::cerr << "Invalid TT_LOG_MAX_SIZE value, using default 10MB" << std::endl;
                }
            }

            const char* max_files_env = std::getenv("TT_LOG_MAX_FILES");
            if (max_files_env) {
                try {
                    max_files_ = std::stoul(max_files_env);
                } catch (const std::exception&) {
                    std::cerr << "Invalid TT_LOG_MAX_FILES value, using default 3" << std::endl;
                }
            }

            // Create directory if it doesn't exist
            std::filesystem::path file_path(log_file_);
            if (file_path.has_parent_path()) {
                std::filesystem::create_directories(file_path.parent_path());
            }

            // Open file for writing
            file_stream_.open(log_file_, std::ios::app);
            if (!file_stream_.is_open()) {
                std::cerr << "Failed to open log file: " << log_file_ << std::endl;
            } else {
                std::cout << "File logging enabled: " << log_file_
                          << " (max_size: " << (max_file_size_ / 1024 / 1024) << "MB, "
                          << "max_files: " << max_files_ << ")" << std::endl;
            }
        }

        log(Level::INFO, "Logger initialized with level: " + level_to_string(level_));

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize logger: " << e.what() << std::endl;
    }
}

Logger::Level Logger::parse_log_level(const std::string& level_str) {
    std::string lower_level = level_str;
    std::transform(lower_level.begin(), lower_level.end(), lower_level.begin(), ::tolower);

    if (lower_level == "trace") return Level::TRACE;
    if (lower_level == "debug") return Level::DEBUG;
    if (lower_level == "info") return Level::INFO;
    if (lower_level == "warn" || lower_level == "warning") return Level::WARN;
    if (lower_level == "error") return Level::ERROR;
    if (lower_level == "critical") return Level::CRITICAL;
    if (lower_level == "off") return Level::OFF;

    std::cerr << "Unknown log level '" << level_str << "', using 'info'" << std::endl;
    return Level::INFO;
}

std::string Logger::level_to_string(Level level) {
    switch (level) {
        case Level::TRACE: return "trace";
        case Level::DEBUG: return "debug";
        case Level::INFO: return "info";
        case Level::WARN: return "warn";
        case Level::ERROR: return "error";
        case Level::CRITICAL: return "critical";
        case Level::OFF: return "off";
        default: return "unknown";
    }
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void Logger::log(Level level, const std::string& message) {
    if (level < level_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::string timestamp = get_timestamp();
    std::string level_str = level_to_string(level);
    std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::toupper);

    std::string log_line = "[" + timestamp + "] [tt-media-server] [" + level_str + "] " + message;

    // Output to console
    if (level >= Level::ERROR) {
        std::cerr << log_line << std::endl;
    } else {
        std::cout << log_line << std::endl;
    }

    // Output to file if configured
    if (file_stream_.is_open()) {
        rotate_file_if_needed();
        file_stream_ << log_line << std::endl;
        file_stream_.flush();
    }
}

void Logger::rotate_file_if_needed() {
    if (!file_stream_.is_open() || log_file_.empty()) {
        return;
    }

    try {
        std::error_code ec;
        size_t file_size = std::filesystem::file_size(log_file_, ec);
        if (ec || file_size < max_file_size_) {
            return;
        }

        // Close current file
        file_stream_.close();

        // Rotate files
        for (size_t i = max_files_ - 1; i > 0; --i) {
            std::string old_file = log_file_ + "." + std::to_string(i);
            std::string new_file = log_file_ + "." + std::to_string(i + 1);

            if (std::filesystem::exists(old_file)) {
                if (i == max_files_ - 1) {
                    std::filesystem::remove(new_file);  // Remove oldest
                }
                std::filesystem::rename(old_file, new_file);
            }
        }

        // Move current log to .1
        if (std::filesystem::exists(log_file_)) {
            std::filesystem::rename(log_file_, log_file_ + ".1");
        }

        // Reopen new file
        file_stream_.open(log_file_, std::ios::app);

    } catch (const std::exception& e) {
        std::cerr << "Error during log rotation: " << e.what() << std::endl;
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout.flush();
    std::cerr.flush();
    if (file_stream_.is_open()) {
        file_stream_.flush();
    }
}

} // namespace tt::utils
