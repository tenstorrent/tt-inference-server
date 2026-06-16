// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared helpers for unit tests.

#pragma once

#include <gtest/gtest.h>
#include <json/json.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "domain/llm/chat_message.hpp"
#include "services/iservice.hpp"
#include "utils/id_generator.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::test {

// ---------------------------------------------------------------------------
// JSON parsing
// ---------------------------------------------------------------------------

inline Json::Value parseJsonString(const std::string& str) {
  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::istringstream iss(str);
  if (!Json::parseFromStream(builder, iss, &root, &errs)) {
    throw std::runtime_error("JSON parse error: " + errs);
  }
  return root;
}

// ---------------------------------------------------------------------------
// Chat message builders
// ---------------------------------------------------------------------------

inline domain::llm::ChatMessage makeChatMessage(std::string role,
                                                std::string content) {
  domain::llm::ChatMessage m;
  m.role = std::move(role);
  m.content = std::move(content);
  return m;
}

// ---------------------------------------------------------------------------
// Prompt and token generation
// ---------------------------------------------------------------------------

inline std::vector<int64_t> makeSequentialPrompt(size_t length) {
  std::vector<int64_t> prompt(length);
  std::iota(prompt.begin(), prompt.end(), 0);
  return prompt;
}

inline std::vector<int> makeSequentialTokens(size_t count) {
  std::vector<int> tokens(count);
  std::iota(tokens.begin(), tokens.end(), 0);
  return tokens;
}

inline uint32_t generateTaskId() { return utils::TaskIDGenerator::generate(); }

// ---------------------------------------------------------------------------
// Config builders
// ---------------------------------------------------------------------------

inline config::LLMConfig makeLLMConfig(
    int numBlocks = 128, int blockSize = 8, int eos = 0,
    std::vector<int64_t> stopTokenIds = {},
    config::ModelRunnerType runnerType =
        config::ModelRunnerType::MOCK_PIPELINE) {
  config::LLMConfig cfg{};
  cfg.runner_type = runnerType;
  cfg.num_blocks = numBlocks;
  cfg.block_size = blockSize;
  cfg.eos_token_id = eos;
  cfg.stop_token_ids = std::move(stopTokenIds);
  return cfg;
}

// ---------------------------------------------------------------------------
// Mock service
// ---------------------------------------------------------------------------

class MockService : public services::IService {
 public:
  explicit MockService(bool modelReady = true, bool workerReady = true,
                       bool workerAlive = true)
      : modelReady_(modelReady),
        workerReady_(workerReady),
        workerAlive_(workerAlive) {}

  static std::shared_ptr<MockService> throwing() {
    auto service = std::make_shared<MockService>(/*modelReady=*/false);
    service->throwOnStatus_ = true;
    return service;
  }

  void start() override {}
  void stop() override {}
  bool isModelReady() const override { return modelReady_; }

  services::SystemStatus getSystemStatus() const override {
    if (throwOnStatus_) {
      throw std::runtime_error("status failed");
    }
    services::SystemStatus status;
    status.modelReady = modelReady_;
    status.queueSize = 0;
    status.maxQueueSize = 1000;
    return status;
  }

  void setModelReady(bool ready) { modelReady_ = ready; }

 private:
  bool modelReady_;
  bool workerReady_;
  bool workerAlive_;
  bool throwOnStatus_ = false;
};

// ---------------------------------------------------------------------------
// Tokenizer test fixture base
// ---------------------------------------------------------------------------

class TokenizerTestFixture : public ::testing::Test {
 protected:
  std::unique_ptr<utils::tokenizers::Tokenizer> tok_;

  void SetUp() override {
    std::string path = config::tokenizerPath();
    if (path.empty()) {
      GTEST_SKIP() << "Tokenizer path not configured";
    }
    tok_ = utils::tokenizers::createTokenizer(config::modelType(), path);
    if (!tok_->isLoaded()) {
      FAIL() << "Failed to load tokenizer from: " << path;
    }
  }

  utils::tokenizers::Tokenizer& tokenizer() { return *tok_; }
};

// ---------------------------------------------------------------------------
// Shared memory helpers
// ---------------------------------------------------------------------------

inline bool shmExists(const std::string& name) {
  return access(("/dev/shm" + name).c_str(), F_OK) == 0;
}

// ---------------------------------------------------------------------------
// Unique resource naming
// ---------------------------------------------------------------------------

inline std::string makeUniqueResourceName(const std::string& base) {
  return base + "_" + std::to_string(getpid());
}

// ---------------------------------------------------------------------------
// Concurrency test utilities
// ---------------------------------------------------------------------------

// Two-phase barrier for TOCTOU race testing.
struct TwoPhaseBarrier {
  std::mutex mu;
  std::condition_variable cv;
  int phase{0};

  void waitForPhase(int p) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [this, p] { return phase >= p; });
  }

  void advance() {
    std::unique_lock<std::mutex> lock(mu);
    ++phase;
    cv.notify_all();
  }

  void reset() {
    std::unique_lock<std::mutex> lock(mu);
    phase = 0;
    cv.notify_all();
  }
};

}  // namespace tt::test
