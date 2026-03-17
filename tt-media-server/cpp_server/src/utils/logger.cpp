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
  const char* logLevelEnv = std::getenv("TT_LOG_LEVEL");
  if (logLevelEnv) {
    level_ = parse_log_level(logLevelEnv);
  }

  // Create spdlog sinks
  std::vector<spdlog::sink_ptr> sinks;

  // Always add console sink
  auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  consoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [tt-media-server] [%l] %v");
  sinks.push_back(consoleSink);

  // Check for file logging configuration
  const char* logFileEnv = std::getenv("TT_LOG_FILE");
  if (logFileEnv && std::strlen(logFileEnv) > 0) {
    std::filesystem::path logPath(logFileEnv);

    // Create directory if it doesn't exist
    if (logPath.has_parent_path()) {
      std::filesystem::create_directories(logPath.parent_path());
    }

    auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        logFileEnv, 1024 * 1024 * 50, 5);  // 50MB, 5 files
    fileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [tt-media-server] [%l] %v");
    sinks.push_back(fileSink);
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

void ZeroOverheadLogger::log_impl(Level level, const char* fmtStr) {
  auto logger = get_logger();
  switch (level) {
    case TRACE:
      logger->trace(fmtStr);
      break;
    case DEBUG:
      logger->debug(fmtStr);
      break;
    case INFO:
      logger->info(fmtStr);
      break;
    case WARN:
      logger->warn(fmtStr);
      break;
    case ERROR:
      logger->error(fmtStr);
      break;
    case CRITICAL:
      logger->critical(fmtStr);
      break;
    case OFF:
      break;
  }
}

ZeroOverheadLogger::Level ZeroOverheadLogger::parse_log_level(
    const std::string& levelStr) {
  std::string lowerLevel = levelStr;
  std::transform(lowerLevel.begin(), lowerLevel.end(), lowerLevel.begin(),
                 ::tolower);

  if (lowerLevel == "trace") return TRACE;
  if (lowerLevel == "debug") return DEBUG;
  if (lowerLevel == "info") return INFO;
  if (lowerLevel == "warn" || lowerLevel == "warning") return WARN;
  if (lowerLevel == "error") return ERROR;
  if (lowerLevel == "critical") return CRITICAL;
  if (lowerLevel == "off") return OFF;

  std::cerr << "Unknown log level '" << levelStr << "', using 'info'"
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
