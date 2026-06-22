// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared test fixture base for integration tests that route through Dynamo.
//
// Provides:
//   - Dynamo frontend availability checking and graceful skip
//   - Helper methods for sending requests via Dynamo frontend
//   - ChatRequest factory pre-configured with the Dynamo model name
//
// Usage:
//   class MyTest : public DynamoTestFixture<MyTest> {
//    protected:
//     static void SetUpTestSuite() {
//       // Call base first - sets dynamoAvailable_ and dynamoConfig_
//       if (!initDynamo()) return;
//       // Your test-specific setup here...
//     }
//   };

#pragma once

#include <gtest/gtest.h>

#include <future>
#include <iostream>
#include <string>

#include "chat_request.hpp"
#include "dynamo_client.hpp"

namespace tt::test {

template <typename Derived>
class DynamoTestFixture : public ::testing::Test {
 protected:
  // Call this at the start of SetUpTestSuite(). Returns true if Dynamo is
  // available and setup should continue, false if tests should be skipped.
  static bool initDynamo(int timeoutSec = 5) {
    dynamoConfig_ = DynamoConfig::fromEnv();
    if (!waitForDynamoFrontend(dynamoConfig_, timeoutSec)) {
      dynamoAvailable_ = false;
      std::cerr << "[" << testName() << "] Dynamo frontend not available at "
                << dynamoConfig_.host << ":" << dynamoConfig_.port << std::endl;
      std::cerr << "  Start with: cd dynamo_frontend && ./deploy.sh "
                   "--local-build"
                << std::endl;
      return false;
    }
    dynamoAvailable_ = true;
    return true;
  }

  // Call this after your server is started to warm up the Dynamo frontend.
  // Returns true if warmup succeeded.
  static bool warmupDynamo() {
    if (!warmupDynamoFrontend(dynamoConfig_)) {
      std::cerr << "[" << testName() << "] Dynamo frontend warmup failed "
                << "(backend may not have registered with etcd)" << std::endl;
      dynamoAvailable_ = false;
      return false;
    }
    return true;
  }

  void SetUp() override {
    if (!dynamoAvailable_) {
      GTEST_SKIP() << "Dynamo frontend not available. Start with: "
                   << "cd dynamo_frontend && ./deploy.sh --local-build";
    }
  }

  // Fire request in background via Dynamo frontend. Returns future for the
  // raw HTTP response.
  static std::future<std::string> asyncRequest(const std::string& body,
                                               int timeoutMs = 30000) {
    return std::async(std::launch::async, [body, timeoutMs] {
      return sendDynamoRequest(dynamoConfig_, body, timeoutMs);
    });
  }

  static std::future<std::string> asyncRequest(const ChatRequest& req,
                                               int timeoutMs = 30000) {
    return asyncRequest(req.toJson(), timeoutMs);
  }

  // Create a ChatRequest pre-configured with the Dynamo model name.
  static ChatRequest chatRequest() {
    return ChatRequest().model(dynamoConfig_.model);
  }

  static const DynamoConfig& dynamoConfig() { return dynamoConfig_; }
  static bool dynamoAvailable() { return dynamoAvailable_; }

  static DynamoConfig dynamoConfig_;
  static bool dynamoAvailable_;

 private:
  static const char* testName() {
    // Use the derived class name for better error messages
    return typeid(Derived).name();
  }
};

// Static member definitions - must be in header for templates
template <typename Derived>
DynamoConfig DynamoTestFixture<Derived>::dynamoConfig_;

template <typename Derived>
bool DynamoTestFixture<Derived>::dynamoAvailable_ = false;

}  // namespace tt::test
