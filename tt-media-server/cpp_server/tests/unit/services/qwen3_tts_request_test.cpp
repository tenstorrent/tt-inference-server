// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// The parent-side half of Qwen3-TTS serving: parsing the optional request
// fields (JSON and multipart), rejecting requests the served release cannot
// take (each is an HTTP 400 before streaming starts), and the preprocessor
// passing raw fields through instead of compiling TTS-2 prompt tokens.

#include <gtest/gtest.h>
#include <json/json.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "domain/tts/tts_types.hpp"
#include "services/qwen3_tts_request_validator.hpp"
#include "services/tts_request_preprocessor.hpp"

namespace {

namespace qwen3 = tt::services::qwen3_tts;
using tt::config::ModelRunnerType;
using tt::config::Qwen3TtsModelSize;
using tt::config::Qwen3TtsRelease;
using tt::domain::tts::TtsRequest;
using tt::domain::tts::VoiceSample;

VoiceSample voiceSample(double seconds, uint32_t rate = 24000,
                        uint16_t channels = 1) {
  VoiceSample sample;
  sample.sampleRateHz = rate;
  sample.channels = channels;
  const auto frames = static_cast<size_t>(seconds * rate);
  sample.wavPcm.resize(frames * channels);
  for (size_t i = 0; i < sample.wavPcm.size(); ++i) {
    sample.wavPcm[i] = static_cast<int16_t>((i % 200) * 100 - 10000);
  }
  return sample;
}

TtsRequest request(const std::string& text = "Hello there.") {
  TtsRequest r(42);
  r.text = text;
  return r;
}

qwen3::RequestRules rules(
    Qwen3TtsRelease release,
    Qwen3TtsModelSize size = Qwen3TtsModelSize::SIZE_1B7) {
  qwen3::RequestRules out;
  out.release = release;
  out.size = size;
  out.maxReferenceSeconds = 30;
  if (release == Qwen3TtsRelease::CUSTOM_VOICE) {
    out.speakers = {"ryan", "vivian"};
  }
  out.languages = {"chinese", "english"};
  return out;
}

/** The HTTP 400 message, or "" when the request is accepted. */
std::string rejection(const TtsRequest& r, const qwen3::RequestRules& rules) {
  try {
    qwen3::validateRequest(r, rules);
    return "";
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
}

// ---- request parsing ------------------------------------------------------

TEST(TtsRequestParsingTest, JsonCarriesTheOptionalVoiceFields) {
  Json::Value json;
  json["text"] = "Hi";
  json["description"] = "warmly";
  json["speaker"] = "ryan";
  json["language"] = "English";
  json["reference_text"] = "what the clip says";
  const auto r = TtsRequest::fromJson(json, 7);
  EXPECT_EQ(r.task_id, 7u);
  EXPECT_EQ(r.text, "Hi");
  EXPECT_EQ(r.description, "warmly");
  EXPECT_EQ(r.speaker, "ryan");
  EXPECT_EQ(r.language, "English");
  EXPECT_EQ(r.referenceText, "what the clip says");
  EXPECT_FALSE(r.voiceSample.has_value());
}

TEST(TtsRequestParsingTest, JsonLeavesAbsentAndNullFieldsUnset) {
  Json::Value json;
  json["text"] = "Hi";
  json["speaker"] = Json::Value::null;
  const auto r = TtsRequest::fromJson(json, 1);
  EXPECT_FALSE(r.description.has_value());
  EXPECT_FALSE(r.speaker.has_value());
  EXPECT_FALSE(r.language.has_value());
  EXPECT_FALSE(r.referenceText.has_value());
}

TEST(TtsRequestParsingTest, JsonRejectsNonStringFields) {
  Json::Value json;
  json["text"] = "Hi";
  json["reference_text"] = 3;
  EXPECT_THROW(TtsRequest::fromJson(json, 1), std::invalid_argument);
  Json::Value noText;
  noText["speaker"] = "ryan";
  EXPECT_THROW(TtsRequest::fromJson(noText, 1), std::invalid_argument);
}

TEST(TtsRequestParsingTest, FormFieldsCarryTheOptionalVoiceFields) {
  const std::unordered_map<std::string, std::string> params = {
      {"text", "Hi"},
      {"description", "warmly"},
      {"speaker", "vivian"},
      {"language", "Chinese"},
      {"reference_text", "what the clip says"}};
  const auto r = TtsRequest::fromFormFields(params, 9);
  EXPECT_EQ(r.task_id, 9u);
  EXPECT_EQ(r.text, "Hi");
  EXPECT_EQ(r.description, "warmly");
  EXPECT_EQ(r.speaker, "vivian");
  EXPECT_EQ(r.language, "Chinese");
  EXPECT_EQ(r.referenceText, "what the clip says");
}

TEST(TtsRequestParsingTest, FormFieldsKeepTheOldContract) {
  const std::unordered_map<std::string, std::string> params = {{"text", "Hi"}};
  const auto r = TtsRequest::fromFormFields(params, 1);
  EXPECT_EQ(r.text, "Hi");
  EXPECT_FALSE(r.description.has_value());
  EXPECT_FALSE(r.speaker.has_value());
  EXPECT_FALSE(r.language.has_value());
  EXPECT_FALSE(r.referenceText.has_value());

  const std::unordered_map<std::string, std::string> noText = {
      {"speaker", "ryan"}};
  EXPECT_THROW(TtsRequest::fromFormFields(noText, 1), std::invalid_argument);
}

// ---- release validation ---------------------------------------------------

TEST(Qwen3TtsValidationTest, CustomVoiceAcceptsASpeaker) {
  auto r = request();
  r.speaker = "Ryan";  // case-insensitive, as the model reads it
  r.language = "english";
  EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE)), "");
  r.description = "Speak slowly.";  // a delivery instruction, 1.7B only
  EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE)), "");
}

TEST(Qwen3TtsValidationTest, CustomVoiceNeedsAKnownSpeaker) {
  auto r = request();
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE))
                .find("'speaker' is required"),
            std::string::npos);
  r.speaker = "   ";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE)), "");
  r.speaker = "nobody";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE))
                .find("unknown speaker"),
            std::string::npos);
}

TEST(Qwen3TtsValidationTest, CustomVoiceWithUnknownSpeakerListTrustsWorker) {
  auto r = request();
  r.speaker = "nobody";
  auto noLists = rules(Qwen3TtsRelease::CUSTOM_VOICE);
  noLists.speakers.clear();
  EXPECT_EQ(rejection(r, noLists), "");
}

TEST(Qwen3TtsValidationTest, CustomVoiceRefusesCloneInputs) {
  auto r = request();
  r.speaker = "ryan";
  r.voiceSample = voiceSample(3.0);
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE)), "");
  r.voiceSample.reset();
  r.referenceText = "transcript";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE)), "");
}

TEST(Qwen3TtsValidationTest, SmallCustomVoiceRefusesAnInstruction) {
  auto r = request();
  r.speaker = "ryan";
  r.description = "Speak slowly.";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE,
                               Qwen3TtsModelSize::SIZE_0B6))
                .find("0.6B"),
            std::string::npos);
}

TEST(Qwen3TtsValidationTest, VoiceDesignNeedsADescription) {
  auto r = request();
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::VOICE_DESIGN))
                .find("'description' is required"),
            std::string::npos);
  r.description = "A calm older man with a slight rasp.";
  EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::VOICE_DESIGN)), "");
}

TEST(Qwen3TtsValidationTest, VoiceDesignRefusesOtherVoiceInputs) {
  auto r = request();
  r.description = "A calm voice.";
  r.speaker = "ryan";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::VOICE_DESIGN)), "");
  r.speaker.reset();
  r.voiceSample = voiceSample(3.0);
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::VOICE_DESIGN)), "");
  r.voiceSample.reset();
  r.referenceText = "transcript";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::VOICE_DESIGN)), "");
}

TEST(Qwen3TtsValidationTest, BaseClonesInContextOrFromTheVoiceAlone) {
  auto r = request();
  r.voiceSample = voiceSample(3.0);
  EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::BASE)), "");  // x-vector
  r.referenceText = "what the clip says";
  EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::BASE)), "");  // in context
}

TEST(Qwen3TtsValidationTest, BaseNeedsTheVoiceFile) {
  auto r = request();
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::BASE)).find("WAV"),
            std::string::npos);
  r.referenceText = "transcript without a clip";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::BASE)), "");
}

TEST(Qwen3TtsValidationTest, BaseRefusesASpeaker) {
  auto r = request();
  r.voiceSample = voiceSample(3.0);
  r.speaker = "ryan";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::BASE)), "");
}

TEST(Qwen3TtsValidationTest, BaseTakesAnInstructionOnlyForVoiceOnlyClones) {
  auto r = request();
  r.voiceSample = voiceSample(3.0);
  r.description = "Whisper.";
  EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::BASE)), "");
  r.referenceText = "what the clip says";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::BASE)).find("in-context"),
            std::string::npos);
  r.referenceText.reset();
  EXPECT_NE(
      rejection(r, rules(Qwen3TtsRelease::BASE, Qwen3TtsModelSize::SIZE_0B6)),
      "");
}

TEST(Qwen3TtsValidationTest, BaseRefusesAnOverlongClip) {
  auto r = request();
  r.voiceSample = voiceSample(31.0, 16000, 2);
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::BASE)).find("at most 30 s"),
            std::string::npos);
  r.voiceSample = voiceSample(29.0, 16000, 2);
  EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::BASE)), "");
}

TEST(Qwen3TtsValidationTest, LanguageIsAutoOrOneTheCheckpointSpeaks) {
  auto r = request();
  r.speaker = "ryan";
  for (const char* ok : {"Auto", "auto", "English", "CHINESE", "  "}) {
    r.language = ok;
    EXPECT_EQ(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE)), "") << ok;
  }
  r.language = "Klingon";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE))
                .find("unsupported language"),
            std::string::npos);
}

TEST(Qwen3TtsValidationTest, TextMustNotBeBlank) {
  auto r = request("   ");
  r.speaker = "ryan";
  EXPECT_NE(rejection(r, rules(Qwen3TtsRelease::CUSTOM_VOICE)), "");
}

// ---- preprocessor -----------------------------------------------------------

tt::config::TtsConfig qwen3Config(Qwen3TtsRelease release) {
  tt::config::TtsConfig cfg;
  cfg.runner_type = ModelRunnerType::TT_QWEN3_TTS;
  cfg.voiceSampleRateHz = 24000;
  cfg.audioSampleRateHz = 24000;
  cfg.qwen3Release = release;
  cfg.qwen3Languages = {"english"};
  cfg.qwen3Speakers = release == Qwen3TtsRelease::CUSTOM_VOICE
                          ? std::vector<std::string>{"ryan"}
                          : std::vector<std::string>{};
  // No TTS-2 tokenizer: a Qwen3 deployment must not need one.
  cfg.tokenizerPath.clear();
  return cfg;
}

TEST(Qwen3TtsPreprocessorTest, PassesRawFieldsWithoutCompilingTtsTwoTokens) {
  const tt::services::TtsRequestPreprocessor preprocessor(
      qwen3Config(Qwen3TtsRelease::CUSTOM_VOICE));
  auto r = request();
  r.speaker = "ryan";
  r.language = "English";
  r.description = "Slowly.";
  const auto task = preprocessor.process(r);
  EXPECT_EQ(task.task_id, 42u);
  EXPECT_EQ(task.text, "Hello there.");
  EXPECT_TRUE(task.promptTokens.empty());
  EXPECT_EQ(task.speaker, "ryan");
  EXPECT_EQ(task.language, "English");
  EXPECT_EQ(task.description, "Slowly.");
  EXPECT_FALSE(task.referenceText.has_value());
  EXPECT_TRUE(task.voiceWavPcm.empty());
}

TEST(Qwen3TtsPreprocessorTest, TtsTwoStillCompilesItsPrompt) {
  // The same text-only request on a TTS-2 deployment goes to the prompt
  // compiler, which needs the tokenizer this config does not have.
  auto cfg = qwen3Config(Qwen3TtsRelease::CUSTOM_VOICE);
  cfg.runner_type = ModelRunnerType::TT_TTS;
  const tt::services::TtsRequestPreprocessor preprocessor(cfg);
  EXPECT_THROW(preprocessor.process(request()), std::runtime_error);
}

TEST(Qwen3TtsPreprocessorTest, RejectsWhatTheReleaseCannotServe) {
  const tt::services::TtsRequestPreprocessor preprocessor(
      qwen3Config(Qwen3TtsRelease::VOICE_DESIGN));
  EXPECT_THROW(preprocessor.process(request()), std::invalid_argument);
}

TEST(Qwen3TtsPreprocessorTest, ResamplesTheCloneClipTo24kMono) {
  const tt::services::TtsRequestPreprocessor preprocessor(
      qwen3Config(Qwen3TtsRelease::BASE));
  auto r = request();
  r.voiceSample = voiceSample(2.0, 16000, 2);
  r.referenceText = "what the clip says";
  r.description = "  ";  // blank travels as absent
  const auto task = preprocessor.process(r);
  EXPECT_EQ(task.voiceWavPcm.size(), 48000u);
  EXPECT_EQ(task.referenceText, "what the clip says");
  EXPECT_FALSE(task.description.has_value());
  EXPECT_TRUE(task.promptTokens.empty());
}

}  // namespace
