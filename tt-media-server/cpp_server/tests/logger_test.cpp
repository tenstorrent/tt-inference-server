// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <gtest/gtest.h>
#include "utils/logger.hpp"
#include <filesystem>
#include <fstream>

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clean up any existing log files
        if (std::filesystem::exists("test_log.txt")) {
            std::filesystem::remove("test_log.txt");
        }
    }

    void TearDown() override {
        // Clean up test log file
        if (std::filesystem::exists("test_log.txt")) {
            std::filesystem::remove("test_log.txt");
        }
    }
};

TEST_F(LoggerTest, SingletonBehavior) {
    auto logger1 = tt::utils::Logger::get_logger();
    auto logger2 = tt::utils::Logger::get_logger();

    // Should return the same instance
    EXPECT_EQ(logger1, logger2);
    EXPECT_NE(logger1, nullptr);
}

TEST_F(LoggerTest, BasicLogging) {
    auto logger = tt::utils::Logger::get_logger();

    // These should not throw
    EXPECT_NO_THROW(logger->log_info("Test info message"));
    EXPECT_NO_THROW(logger->log_debug("Test debug message"));
    EXPECT_NO_THROW(logger->log_warning("Test warning message"));
    EXPECT_NO_THROW(logger->log_error("Test error message"));
}

TEST_F(LoggerTest, FormattedLogging) {
    auto logger = tt::utils::Logger::get_logger();

    // Test formatted logging
    EXPECT_NO_THROW(logger->log_info("Test message with number: {}", 42));
    EXPECT_NO_THROW(logger->log_error("Error with string: {} and number: {}", "test", 123));
}

TEST_F(LoggerTest, ConvenienceMacros) {
    // Test the convenience macros
    EXPECT_NO_THROW(TT_LOG_INFO("Macro test info"));
    EXPECT_NO_THROW(TT_LOG_DEBUG("Macro test debug"));
    EXPECT_NO_THROW(TT_LOG_WARN("Macro test warning"));
    EXPECT_NO_THROW(TT_LOG_ERROR("Macro test error"));
    EXPECT_NO_THROW(TT_LOG_INFO("Macro test with format: {}", 42));
}

TEST_F(LoggerTest, LogLevel) {
    auto logger = tt::utils::Logger::get_logger();

    // Test setting log level
    EXPECT_NO_THROW(logger->set_level(tt::utils::Logger::Level::DEBUG));
    EXPECT_NO_THROW(logger->set_level(tt::utils::Logger::Level::INFO));
    EXPECT_NO_THROW(logger->set_level(tt::utils::Logger::Level::WARN));
}

TEST_F(LoggerTest, Flush) {
    auto logger = tt::utils::Logger::get_logger();

    logger->log_info("Test message before flush");
    EXPECT_NO_THROW(logger->flush());
}

TEST_F(LoggerTest, StringFormatting) {
    auto logger = tt::utils::Logger::get_logger();

    // Test different types of arguments
    EXPECT_NO_THROW(logger->log_info("String: {}, Int: {}, Float: {}", "hello", 42, 3.14));
    EXPECT_NO_THROW(logger->log_info("Multiple strings: {} {}", "first", "second"));
    EXPECT_NO_THROW(logger->log_info("No format arguments"));
}
