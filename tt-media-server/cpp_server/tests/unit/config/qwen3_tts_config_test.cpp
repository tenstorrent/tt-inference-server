// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Which Qwen3-TTS release a deployment serves: parsing, reading a checkpoint's
// config.json, finding it the way tt-metal's weights.checkpoint_dir() would,
// and the precedence between explicit settings, the file and the hub id.

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/qwen3_tts.hpp"

namespace {

namespace fs = std::filesystem;
namespace qwen3 = tt::config::qwen3_tts;
using tt::config::Qwen3TtsModelSize;
using tt::config::Qwen3TtsRelease;

constexpr const char* CUSTOM_VOICE_CONFIG = R"({
  "tts_model_type": "custom_voice",
  "tts_model_size": "0b6",
  "talker_config": {
    "spk_id": {"Ryan": 3000, "vivian": 3001},
    "codec_language_id": {"english": 2050, "Chinese": 2055}
  }
})";

constexpr const char* BASE_CONFIG = R"({
  "tts_model_type": "base",
  "tts_model_size": "1b7",
  "talker_config": {"spk_id": {}, "codec_language_id": {"english": 2050}}
})";

/** A scratch directory removed with the test. */
class TempDir {
 public:
  TempDir() {
    path = fs::temp_directory_path() /
           ("qwen3_tts_config_test_" + std::to_string(::getpid()) + "_" +
            std::to_string(counter++));
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }

  fs::path writeFile(const fs::path& relative, const std::string& content) {
    const fs::path file = path / relative;
    fs::create_directories(file.parent_path());
    std::ofstream(file) << content;
    return file;
  }

  fs::path path;

 private:
  static inline int counter = 0;
};

TEST(Qwen3TtsConfigTest, ParsesReleaseAndSize) {
  EXPECT_EQ(qwen3::releaseFromString("base"), Qwen3TtsRelease::BASE);
  EXPECT_EQ(qwen3::releaseFromString(" Custom_Voice "),
            Qwen3TtsRelease::CUSTOM_VOICE);
  EXPECT_EQ(qwen3::releaseFromString("voice_design"),
            Qwen3TtsRelease::VOICE_DESIGN);
  EXPECT_FALSE(qwen3::releaseFromString("customvoice").has_value());
  EXPECT_FALSE(qwen3::releaseFromString("").has_value());

  EXPECT_EQ(qwen3::modelSizeFromString("1b7"), Qwen3TtsModelSize::SIZE_1B7);
  EXPECT_EQ(qwen3::modelSizeFromString("0.6B"), Qwen3TtsModelSize::SIZE_0B6);
  EXPECT_FALSE(qwen3::modelSizeFromString("7b").has_value());
}

TEST(Qwen3TtsConfigTest, StringsMatchTheModelsOwnNames) {
  // weights.model_kind() / model_size() return these; the worker compares.
  EXPECT_EQ(tt::config::toString(Qwen3TtsRelease::BASE), "base");
  EXPECT_EQ(tt::config::toString(Qwen3TtsRelease::CUSTOM_VOICE),
            "custom_voice");
  EXPECT_EQ(tt::config::toString(Qwen3TtsRelease::VOICE_DESIGN),
            "voice_design");
  EXPECT_EQ(tt::config::toString(Qwen3TtsModelSize::SIZE_1B7), "1b7");
  EXPECT_EQ(tt::config::toString(Qwen3TtsModelSize::SIZE_0B6), "0b6");
}

TEST(Qwen3TtsConfigTest, GuessesReleaseAndSizeFromHubIds) {
  EXPECT_EQ(qwen3::releaseFromModelName("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            Qwen3TtsRelease::BASE);
  EXPECT_EQ(qwen3::releaseFromModelName("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            Qwen3TtsRelease::CUSTOM_VOICE);
  EXPECT_EQ(qwen3::releaseFromModelName("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
            Qwen3TtsRelease::VOICE_DESIGN);
  EXPECT_FALSE(qwen3::releaseFromModelName("my-org/finetune").has_value());

  EXPECT_EQ(qwen3::modelSizeFromModelName("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            Qwen3TtsModelSize::SIZE_0B6);
  EXPECT_EQ(qwen3::modelSizeFromModelName("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            Qwen3TtsModelSize::SIZE_1B7);
  EXPECT_FALSE(qwen3::modelSizeFromModelName("my-org/finetune").has_value());
}

TEST(Qwen3TtsConfigTest, ReadsCheckpointConfig) {
  TempDir dir;
  const auto file = dir.writeFile("config.json", CUSTOM_VOICE_CONFIG);
  const auto info = qwen3::readCheckpointConfig(file.string());
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->release, Qwen3TtsRelease::CUSTOM_VOICE);
  EXPECT_EQ(info->size, Qwen3TtsModelSize::SIZE_0B6);
  EXPECT_EQ(info->speakers, (std::vector<std::string>{"ryan", "vivian"}));
  EXPECT_EQ(info->languages, (std::vector<std::string>{"chinese", "english"}));
  EXPECT_EQ(info->configPath, file.string());
}

TEST(Qwen3TtsConfigTest, MissingOrBrokenConfigReadsAsNothing) {
  TempDir dir;
  EXPECT_FALSE(qwen3::readCheckpointConfig("").has_value());
  EXPECT_FALSE(qwen3::readCheckpointConfig((dir.path / "absent.json").string())
                   .has_value());
  const auto broken = dir.writeFile("config.json", "{ not json");
  EXPECT_FALSE(qwen3::readCheckpointConfig(broken.string()).has_value());
}

TEST(Qwen3TtsConfigTest, FindsConfigInCheckpointDirFirst) {
  TempDir ckpt;
  TempDir hub;
  const auto file = ckpt.writeFile("config.json", BASE_CONFIG);
  hub.writeFile(
      "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/abc/"
      "config.json",
      CUSTOM_VOICE_CONFIG);
  EXPECT_EQ(qwen3::findCheckpointConfig(ckpt.path.string(),
                                        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                        hub.path.string()),
            file.string());
}

TEST(Qwen3TtsConfigTest, CheckpointDirWithoutConfigFindsNothing) {
  // weights.checkpoint_dir() stops at QWEN3_TTS_CKPT; so does the lookup.
  TempDir ckpt;
  TempDir hub;
  hub.writeFile(
      "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/abc/"
      "config.json",
      BASE_CONFIG);
  EXPECT_EQ(
      qwen3::findCheckpointConfig(ckpt.path.string(), "", hub.path.string()),
      "");
}

TEST(Qwen3TtsConfigTest, FindsConfigInHfModelDirectory) {
  TempDir model;
  const auto file = model.writeFile("config.json", BASE_CONFIG);
  EXPECT_EQ(qwen3::findCheckpointConfig("", model.path.string(), ""),
            file.string());
}

TEST(Qwen3TtsConfigTest, FindsConfigInHubCache) {
  TempDir hub;
  const auto file = hub.writeFile(
      "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/rev/"
      "config.json",
      CUSTOM_VOICE_CONFIG);
  EXPECT_EQ(qwen3::findCheckpointConfig(
                "", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", hub.path.string()),
            file.string());
  // Not downloaded yet.
  EXPECT_EQ(qwen3::findCheckpointConfig(
                "", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", hub.path.string()),
            "");
}

TEST(Qwen3TtsConfigTest, NoSourceMeansTheDefaultBaseRepo) {
  TempDir hub;
  const auto file = hub.writeFile(
      "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/rev/config.json",
      BASE_CONFIG);
  EXPECT_EQ(qwen3::findCheckpointConfig("", "", hub.path.string()),
            file.string());

  const auto resolved = qwen3::resolveRelease({});
  EXPECT_EQ(resolved.release, Qwen3TtsRelease::BASE);
  EXPECT_EQ(resolved.size, Qwen3TtsModelSize::SIZE_1B7);
}

TEST(Qwen3TtsConfigTest, ResolvesFromHubIdWhenNotDownloaded) {
  const auto resolved = qwen3::resolveRelease(
      {.hfModel = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"});
  EXPECT_EQ(resolved.release, Qwen3TtsRelease::CUSTOM_VOICE);
  EXPECT_EQ(resolved.size, Qwen3TtsModelSize::SIZE_0B6);
}

TEST(Qwen3TtsConfigTest, CheckpointConfigBeatsTheName) {
  qwen3::CheckpointInfo info;
  info.release = Qwen3TtsRelease::VOICE_DESIGN;
  info.size = Qwen3TtsModelSize::SIZE_1B7;
  info.configPath = "/ckpt/config.json";
  const auto resolved = qwen3::resolveRelease(
      {.hfModel = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", .checkpoint = info});
  EXPECT_EQ(resolved.release, Qwen3TtsRelease::VOICE_DESIGN);
  EXPECT_EQ(resolved.size, Qwen3TtsModelSize::SIZE_1B7);
}

TEST(Qwen3TtsConfigTest, ExplicitSettingsBeatTheName) {
  const auto resolved =
      qwen3::resolveRelease({.explicitRelease = "custom_voice",
                             .explicitSize = "0b6",
                             .hfModel = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"});
  EXPECT_EQ(resolved.release, Qwen3TtsRelease::CUSTOM_VOICE);
  EXPECT_EQ(resolved.size, Qwen3TtsModelSize::SIZE_0B6);
}

TEST(Qwen3TtsConfigTest, LocalDirectoryNeedsConfigOrExplicitRelease) {
  // A directory can be called anything; its name is not evidence.
  EXPECT_THROW(qwen3::resolveRelease({.ckptDir = "/models/my-Base-copy"}),
               std::runtime_error);
  const auto resolved = qwen3::resolveRelease({.explicitRelease = "base",
                                               .explicitSize = "1b7",
                                               .ckptDir = "/models/x"});
  EXPECT_EQ(resolved.release, Qwen3TtsRelease::BASE);
}

TEST(Qwen3TtsConfigTest, RejectsExplicitValuesTheCheckpointContradicts) {
  qwen3::CheckpointInfo info;
  info.release = Qwen3TtsRelease::BASE;
  info.size = Qwen3TtsModelSize::SIZE_1B7;
  info.configPath = "/ckpt/config.json";
  EXPECT_THROW(qwen3::resolveRelease({.explicitRelease = "voice_design",
                                      .ckptDir = "/ckpt",
                                      .checkpoint = info}),
               std::runtime_error);
  EXPECT_THROW(
      qwen3::resolveRelease(
          {.explicitSize = "0b6", .ckptDir = "/ckpt", .checkpoint = info}),
      std::runtime_error);
}

TEST(Qwen3TtsConfigTest, RejectsUnknownExplicitValues) {
  EXPECT_THROW(qwen3::resolveRelease({.explicitRelease = "clone"}),
               std::runtime_error);
  EXPECT_THROW(qwen3::resolveRelease({.explicitSize = "7b"}),
               std::runtime_error);
}

TEST(Qwen3TtsConfigTest, RejectsTheReleaseThatDoesNotExist) {
  EXPECT_THROW(qwen3::resolveRelease(
                   {.hfModel = "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign"}),
               std::runtime_error);
}

}  // namespace
