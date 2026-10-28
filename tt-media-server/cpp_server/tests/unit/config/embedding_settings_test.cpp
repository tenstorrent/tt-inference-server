// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// embeddingEngineConfig() is cached per process, so each case runs in a
// death-test child.

#include <gtest/gtest.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <variant>
#include <vector>

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "config/types.hpp"

namespace {

using tt::config::ModelRunnerType;

void baseEnv(const char* runner, const char* device) {
  ::setenv("MODEL_SERVICE", "embedding", 1);
  ::setenv("MODEL_RUNNER_TYPE", runner, 1);
  ::setenv("DEVICE", device, 1);
  ::setenv("DEVICE_IDS", "(0)", 1);
}

/** Exit 0 when `check` holds, 1 when it does not, 2 when the config threw. */
template <typename Check>
[[noreturn]] void runChild(Check check) {
  try {
    std::_Exit(check() ? 0 : 1);
  } catch (const std::exception& e) {
    std::cerr << "config threw: " << e.what() << std::endl;
    std::_Exit(2);
  }
}

class EmbeddingSettingsTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(EmbeddingSettingsTest, RunnerTypeStringRoundTrips) {
  EXPECT_EQ(tt::config::toString(ModelRunnerType::TT_EMBEDDING_GEMMA2),
            "tt_embedding_gemma2");
}

TEST_F(EmbeddingSettingsTest, EmbeddingGemma2OnP150) {
  EXPECT_EXIT(
      {
        baseEnv("tt_embedding_gemma2", "p150");
        runChild([] {
          const auto cfg = tt::config::embeddingEngineConfig();
          return cfg.runner_type == ModelRunnerType::TT_EMBEDDING_GEMMA2 &&
                 cfg.hf_model_id == "google/embeddinggemma-2" &&
                 cfg.max_batch_size == 32 && cfg.max_seq_len == 8192 &&
                 cfg.mesh_shape == std::vector<size_t>{1, 1} &&
                 cfg.num_command_queues == 0 &&
                 cfg.trace_region_size == 512u * 1024 * 1024 &&
                 cfg.visible_devices == "0";
        });
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(EmbeddingSettingsTest, EmbeddingGemma2WorkerConfig) {
  EXPECT_EXIT(
      {
        baseEnv("tt_embedding_gemma2", "p150");
        runChild([] {
          const auto cfg = std::get<tt::config::EmbeddingConfig>(
              tt::config::workerRunnerConfig(0));
          return cfg.worker_id == 0 && cfg.visible_devices == "0" &&
                 cfg.hf_model_id == "google/embeddinggemma-2";
        });
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(EmbeddingSettingsTest, EmbeddingGemma2RefusesWormhole) {
  EXPECT_EXIT(
      {
        baseEnv("tt_embedding_gemma2", "n150");
        runChild([] {
          tt::config::embeddingEngineConfig();
          return true;
        });
      },
      ::testing::ExitedWithCode(2), "is not supported by tt_embedding_gemma2");
}

}  // namespace
