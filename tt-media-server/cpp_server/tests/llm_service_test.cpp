// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "domain/llm_request.hpp"
#include "services/llm_service.hpp"

namespace tt::services {

class LLMServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up environment for mock backend
    setenv("LLM_DEVICE_BACKEND", "mock", 1);
  }
};

// Test: Valid stop sequences are accepted
TEST_F(LLMServiceTest, ValidStopSequences) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {"\n", "END"};

  EXPECT_NO_THROW(service.preProcess(request));
}

// Test: Single stop sequence is valid
TEST_F(LLMServiceTest, SingleStopSequence) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {"STOP"};

  EXPECT_NO_THROW(service.preProcess(request));
}

// Test: Empty stop array is valid
TEST_F(LLMServiceTest, EmptyStopArray) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {};

  EXPECT_NO_THROW(service.preProcess(request));
}

// Test: Maximum allowed stop sequences (4)
TEST_F(LLMServiceTest, MaxStopSequences) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {"STOP1", "STOP2", "STOP3", "STOP4"};

  EXPECT_NO_THROW(service.preProcess(request));
}

// Test: Too many stop sequences should throw
TEST_F(LLMServiceTest, TooManyStopSequences) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {"STOP1", "STOP2", "STOP3", "STOP4", "STOP5"};

  EXPECT_THROW(
      {
        try {
          service.preProcess(request);
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ(
              "Too many stop sequences: 5 exceeds maximum of 4",
              e.what());
          throw;
        }
      },
      std::invalid_argument);
}

// Test: Empty string in stop sequences should throw
TEST_F(LLMServiceTest, EmptyStopString) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {"STOP1", "", "STOP3"};

  EXPECT_THROW(
      {
        try {
          service.preProcess(request);
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ(
              "Stop sequence at index 1 cannot be empty",
              e.what());
          throw;
        }
      },
      std::invalid_argument);
}

// Test: Multi-character stop sequences are valid
TEST_F(LLMServiceTest, MultiCharacterStopSequences) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {"<|endoftext|>", "###", "\n\n"};

  EXPECT_NO_THROW(service.preProcess(request));
}

// Test: Special characters in stop sequences are valid
TEST_F(LLMServiceTest, SpecialCharactersInStopSequences) {
  LLMService service;

  domain::LLMRequest request{1};  // task_id = 1
  request.prompt = "Test prompt";
  request.stop = {"\n", "\t", "\r\n", "."};

  EXPECT_NO_THROW(service.preProcess(request));
}

}  // namespace tt::services
