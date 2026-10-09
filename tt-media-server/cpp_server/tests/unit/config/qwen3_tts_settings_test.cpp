// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// ttsEngineConfig() / workerRunnerConfig() for MODEL_RUNNER_TYPE=tt_qwen3_tts.
// Both cache their answer for the life of the process, so each case runs in a
// fresh child via a "threadsafe" death test (the child re-executes this binary
// filtered to that one test) and reports through its exit code.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <variant>

#include "config/defaults.hpp"
#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "config/types.hpp"

namespace {

using tt::config::ModelRunnerType;
using tt::config::Qwen3TtsModelSize;
using tt::config::Qwen3TtsRelease;

/** Environment shared by every case: TTS service, two workers, and a hub
 *  cache that holds nothing, so only the hub id can name the release. */
void baseEnv() {
  ::setenv("MODEL_SERVICE", "tts", 1);
  ::setenv("MODEL_RUNNER_TYPE", "tt_qwen3_tts", 1);
  ::setenv("DEVICE_IDS", "(0),(1)", 1);
  const auto emptyHub =
      std::filesystem::temp_directory_path() /
      ("qwen3_tts_settings_test_hub_" + std::to_string(::getpid()));
  ::setenv("HF_HUB_CACHE", emptyHub.c_str(), 1);
  for (const char* name :
       {"QWEN3_TTS_CKPT", "HF_MODEL", "TTS_QWEN3_RELEASE",
        "TTS_QWEN3_MODEL_SIZE", "TTS_QWEN3_SEED", "TTS_MAX_USERS",
        "TTS_AUDIO_SAMPLE_RATE_HZ", "TTS_VOICE_SAMPLE_RATE_HZ",
        "TTS_TOKENIZER_PATH", "TTS_QWEN3_MAX_FRAMES",
        "TTS_QWEN3_MAX_REFERENCE_SECONDS"}) {
    ::unsetenv(name);
  }
}

/** Run `check` in this (child) process: exit 0 when it holds, 1 when it does
 *  not, 2 when building the config threw. */
template <typename Check>
[[noreturn]] void runChild(Check check) {
  try {
    std::_Exit(check() ? 0 : 1);
  } catch (const std::exception& e) {
    std::cerr << "config threw: " << e.what() << std::endl;
    std::_Exit(2);
  }
}

class Qwen3TtsSettingsTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(Qwen3TtsSettingsTest, DerivesReleaseAndQwen3Defaults) {
  EXPECT_EXIT(
      {
        baseEnv();
        ::setenv("HF_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 1);
        ::setenv("TTS_QWEN3_SEED", "7", 1);
        runChild([] {
          const auto cfg = tt::config::ttsEngineConfig();
          return cfg.runner_type == ModelRunnerType::TT_QWEN3_TTS &&
                 cfg.qwen3Release == Qwen3TtsRelease::CUSTOM_VOICE &&
                 cfg.qwen3ModelSize == Qwen3TtsModelSize::SIZE_1B7 &&
                 cfg.audioSampleRateHz == 24000 &&
                 cfg.voiceSampleRateHz == 24000 && cfg.maxUsers == 2 &&
                 cfg.maxBatchSize == 1 && cfg.tokenizerPath.empty() &&
                 cfg.bosToken.empty() && cfg.qwen3MaxFrames == 400 &&
                 cfg.qwen3Seed == 7u &&
                 cfg.qwen3HfModel == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
        });
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(Qwen3TtsSettingsTest, WorkersGetTheirOwnDevices) {
  EXPECT_EXIT(
      {
        baseEnv();
        runChild([] {
          const auto cfg = std::get<tt::config::TtsConfig>(
              tt::config::workerRunnerConfig(1));
          return cfg.workerId == 1 && cfg.visibleDevices == "1" &&
                 cfg.qwen3Release == Qwen3TtsRelease::BASE;
        });
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(Qwen3TtsSettingsTest, ExplicitMaxUsersWins) {
  EXPECT_EXIT(
      {
        baseEnv();
        ::setenv("TTS_MAX_USERS", "5", 1);
        runChild([] { return tt::config::ttsEngineConfig().maxUsers == 5; });
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(Qwen3TtsSettingsTest, RefusesAnotherOutputRate) {
  EXPECT_EXIT(
      {
        baseEnv();
        ::setenv("TTS_AUDIO_SAMPLE_RATE_HZ", "48000", 1);
        runChild([] {
          tt::config::ttsEngineConfig();
          return true;
        });
      },
      ::testing::ExitedWithCode(2), "TTS_AUDIO_SAMPLE_RATE_HZ");
}

TEST_F(Qwen3TtsSettingsTest, AcceptsTheRateItWouldPickAnyway) {
  EXPECT_EXIT(
      {
        baseEnv();
        ::setenv("TTS_AUDIO_SAMPLE_RATE_HZ", "24000", 1);
        runChild([] {
          return tt::config::ttsEngineConfig().audioSampleRateHz == 24000;
        });
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(Qwen3TtsSettingsTest, RefusesAnUnknowableRelease) {
  EXPECT_EXIT(
      {
        baseEnv();
        ::setenv("HF_MODEL", "my-org/qwen3-tts-finetune", 1);
        runChild([] {
          tt::config::ttsEngineConfig();
          return true;
        });
      },
      ::testing::ExitedWithCode(2), "TTS_QWEN3_RELEASE");
}

TEST_F(Qwen3TtsSettingsTest, TtsTwoKeepsItsDefaults) {
  EXPECT_EXIT(
      {
        baseEnv();
        ::setenv("MODEL_RUNNER_TYPE", "mock_tts", 1);
        ::setenv("TTS_TOKENIZER_PATH", "/nonexistent/tokenizer.json", 1);
        runChild([] {
          const auto cfg = tt::config::ttsEngineConfig();
          return cfg.runner_type == ModelRunnerType::MOCK_SCHEDULER &&
                 cfg.audioSampleRateHz == 48000 &&
                 cfg.voiceSampleRateHz == 16000 &&
                 cfg.tokenizerPath == "/nonexistent/tokenizer.json" &&
                 cfg.maxUsers == tt::config::defaults::PM_MAX_USERS;
        });
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(Qwen3TtsRunnerTypeTest, Names) {
  EXPECT_EQ(tt::config::toString(ModelRunnerType::TT_QWEN3_TTS),
            "tt_qwen3_tts");
  EXPECT_EQ(tt::config::toClientRunnerName(ModelRunnerType::TT_QWEN3_TTS),
            "tt-qwen3-tts");
}

}  // namespace
