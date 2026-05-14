// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "utils/logger.hpp"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "config/defaults.hpp"

namespace tt::utils {

ZeroOverheadLogger::Level ZeroOverheadLogger::level = ZeroOverheadLogger::INFO;
std::shared_ptr<spdlog::logger> ZeroOverheadLogger::logger;
bool ZeroOverheadLogger::initialized = false;

void ZeroOverheadLogger::initialize() {
  if (initialized) {
    return;
  }

  // Set log level from environment variable
  const char* logLevelEnv = std::getenv("TT_LOG_LEVEL");
  if (logLevelEnv) {
    level = parseLogLevel(logLevelEnv);
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
        logFileEnv, tt::config::defaults::LOG_FILE_MAX_BYTES,
        tt::config::defaults::LOG_FILE_MAX_COUNT);
    fileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [tt-media-server] [%l] %v");
    sinks.push_back(fileSink);
  }

  // Create logger
  logger = std::make_shared<spdlog::logger>("tt-media-server", sinks.begin(),
                                            sinks.end());

  // Set spdlog level
  switch (level) {
    case TRACE:
      logger->set_level(spdlog::level::trace);
      break;
    case DEBUG:
      logger->set_level(spdlog::level::debug);
      break;
    case INFO:
      logger->set_level(spdlog::level::info);
      break;
    case WARN:
      logger->set_level(spdlog::level::warn);
      break;
    case ERROR:
      logger->set_level(spdlog::level::err);
      break;
    case CRITICAL:
      logger->set_level(spdlog::level::critical);
      break;
    case OFF:
      logger->set_level(spdlog::level::off);
      break;
  }

  // Register logger globally
  spdlog::register_logger(logger);

  initialized = true;

  // Log initialization message
  logger->info("Logger initialized with level: {}", levelToString(level));
}

std::shared_ptr<spdlog::logger> ZeroOverheadLogger::getLogger() {
  if (!initialized) {
    initialize();
  }
  return logger;
}

void ZeroOverheadLogger::logImpl(Level level, const char* fmtStr) {
  auto logger = getLogger();
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

ZeroOverheadLogger::Level ZeroOverheadLogger::parseLogLevel(
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

std::string ZeroOverheadLogger::levelToString(Level level) {
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

}  // namespace tt::utils
