// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "utils/logger.hpp"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace tt::utils {

ZeroOverheadLogger::Level ZeroOverheadLogger::level_ = ZeroOverheadLogger::INFO;
std::shared_ptr<spdlog::logger> ZeroOverheadLogger::logger_;
bool ZeroOverheadLogger::initialized_ = false;

void ZeroOverheadLogger::initialize() {
  if (initialized_) {
    return;
  }

  // Set log level from environment variable
  const char* log_level_env = std::getenv("TT_LOG_LEVEL");
  if (log_level_env) {
    level_ = parse_log_level(log_level_env);
  }

  // Create spdlog sinks
  std::vector<spdlog::sink_ptr> sinks;

  // Always add console sink
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [tt-media-server] [%l] %v");
  sinks.push_back(console_sink);

  // Check for file logging configuration
  const char* log_file_env = std::getenv("TT_LOG_FILE");
  if (log_file_env && std::strlen(log_file_env) > 0) {
    std::filesystem::path log_path(log_file_env);

    // Create directory if it doesn't exist
    if (log_path.has_parent_path()) {
      std::filesystem::create_directories(log_path.parent_path());
    }

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file_env, 1024 * 1024 * 50, 5);  // 50MB, 5 files
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [tt-media-server] [%l] %v");
    sinks.push_back(file_sink);
  }

  // Create logger
  logger_ = std::make_shared<spdlog::logger>("tt-media-server", sinks.begin(),
                                             sinks.end());

  // Set spdlog level
  switch (level_) {
    case TRACE:
      logger_->set_level(spdlog::level::trace);
      break;
    case DEBUG:
      logger_->set_level(spdlog::level::debug);
      break;
    case INFO:
      logger_->set_level(spdlog::level::info);
      break;
    case WARN:
      logger_->set_level(spdlog::level::warn);
      break;
    case ERROR:
      logger_->set_level(spdlog::level::err);
      break;
    case CRITICAL:
      logger_->set_level(spdlog::level::critical);
      break;
    case OFF:
      logger_->set_level(spdlog::level::off);
      break;
  }

  // Register logger globally
  spdlog::register_logger(logger_);

  initialized_ = true;

  // Log initialization message
  logger_->info("Logger initialized with level: {}", level_to_string(level_));
}

std::shared_ptr<spdlog::logger> ZeroOverheadLogger::get_logger() {
  if (!initialized_) {
    initialize();
  }
  return logger_;
}

void ZeroOverheadLogger::log_impl(Level level, const char* fmt_str) {
  auto logger = get_logger();
  switch (level) {
    case TRACE:
      logger->trace(fmt_str);
      break;
    case DEBUG:
      logger->debug(fmt_str);
      break;
    case INFO:
      logger->info(fmt_str);
      break;
    case WARN:
      logger->warn(fmt_str);
      break;
    case ERROR:
      logger->error(fmt_str);
      break;
    case CRITICAL:
      logger->critical(fmt_str);
      break;
    case OFF:
      break;
  }
}

ZeroOverheadLogger::Level ZeroOverheadLogger::parse_log_level(
    const std::string& level_str) {
  std::string lower_level = level_str;
  std::transform(lower_level.begin(), lower_level.end(), lower_level.begin(),
                 ::tolower);

  if (lower_level == "trace") return TRACE;
  if (lower_level == "debug") return DEBUG;
  if (lower_level == "info") return INFO;
  if (lower_level == "warn" || lower_level == "warning") return WARN;
  if (lower_level == "error") return ERROR;
  if (lower_level == "critical") return CRITICAL;
  if (lower_level == "off") return OFF;

  std::cerr << "Unknown log level '" << level_str << "', using 'info'"
            << std::endl;
  return INFO;
}

std::string ZeroOverheadLogger::level_to_string(Level level) {
  switch (level) {
    case TRACE:
      return "TRACE";
    case DEBUG:
      return "DEBUG";
    case INFO:
      return "INFO";
    case WARN:
      return "WARN";
    case ERROR:
      return "ERROR";
    case CRITICAL:
      return "CRITICAL";
    case OFF:
      return "OFF";
    default:
      return "UNKNOWN";
  }
}

void initialize_logger() { ZeroOverheadLogger::initialize(); }

}  // namespace tt::utils
