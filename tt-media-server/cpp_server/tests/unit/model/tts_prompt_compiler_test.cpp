// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tts_prompt_compiler.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <optional>
#include <string>
#include <vector>

#include "domain/tts/tts_types.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace {

namespace compiler = tt::utils::tts_prompt_compiler;

// <|reserved_token_0|>..<|reserved_token_7|>, as the reference compiler emits
// ahead of every <|speech_start|>.
const std::string kReadout =
    "<|reserved_token_0|><|reserved_token_1|><|reserved_token_2|>"
    "<|reserved_token_3|><|reserved_token_4|><|reserved_token_5|>"
    "<|reserved_token_6|><|reserved_token_7|>";

TEST(TtsPromptCompilerTest, CompilesTextOnlyPrompt) {
  EXPECT_EQ(compiler::compilePromptString("  hello there  ", std::nullopt),
            "<|bot|>hello there" + kReadout + "<|speech_start|>");
}

TEST(TtsPromptCompilerTest, CompilesDescriptionPrompt) {
  EXPECT_EQ(
      compiler::compilePromptString("hello", std::string("  calm voice ")),
      "<|voice_prompt_start|>calm voice<|voice_prompt_end|>"
      "<|bot|>hello" +
          kReadout + "<|speech_start|>");
}

// The reference lowercases the voice prompt (tests/test_prompting.py::
// test_voice_prompt_is_lowercased).
TEST(TtsPromptCompilerTest, LowercasesDescription) {
  EXPECT_EQ(compiler::compilePromptString("Hi.", std::string("A Calm Voice")),
            "<|voice_prompt_start|>a calm voice<|voice_prompt_end|>"
            "<|bot|>Hi." +
                kReadout + "<|speech_start|>");
}

// Mirrors tests/test_prompting.py::test_single_turn_layout in the reference.
TEST(TtsPromptCompilerTest, MatchesReferenceSingleTurnLayout) {
  const std::vector<uint32_t> speechIds = {5, 6};
  EXPECT_EQ(compiler::compilePromptString("Hi there.", std::nullopt, speechIds),
            "<|audio_prompt_start|><|s_5|><|s_6|><|audio_prompt_end|>"
            "<|bot|>Hi there." +
                kReadout + "<|speech_start|>");
}

TEST(TtsPromptCompilerTest, CompilesVoiceSamplePrompt) {
  const std::vector<uint32_t> speechIds = {12, 34, 56};
  EXPECT_EQ(compiler::compilePromptString("hello", std::string("calm voice"),
                                          speechIds),
            "<|audio_prompt_start|><|s_12|><|s_34|><|s_56|>"
            "<|audio_prompt_end|><|voice_prompt_start|>calm voice"
            "<|voice_prompt_end|><|bot|>hello" +
                kReadout + "<|speech_start|>");
}

TEST(TtsPromptCompilerTest, AcceptsLastSpeechIdInVocabulary) {
  EXPECT_NO_THROW(compiler::compilePromptString("hello", std::nullopt,
                                                {0, 65535}));
}

TEST(TtsPromptCompilerTest, RejectsSpeechIdOutsideVocabulary) {
  EXPECT_THROW(compiler::compilePromptString("hello", std::nullopt, {65536}),
               std::invalid_argument);
}

TEST(TtsPromptCompilerTest, RejectsEmptyText) {
  EXPECT_THROW(compiler::compilePromptString("   ", std::nullopt),
               std::invalid_argument);
}

TEST(TtsPromptCompilerTest, RejectsEmptyDescription) {
  EXPECT_THROW(compiler::compilePromptString("hello", std::string("   ")),
               std::invalid_argument);
}

TEST(TtsPromptCompilerTest, TokenizesCompiledPromptString) {
  const std::string text = "hello";
  const std::optional<std::string> description = std::string("calm voice");
  const std::vector<uint32_t> speechIds = {12, 34, 56};
  const std::string prompt =
      compiler::compilePromptString(text, description, speechIds);
  const auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  EXPECT_EQ(
      compiler::compilePromptTokens(tokenizer, text, description, speechIds),
      tokenizer.encode(prompt));
}

TEST(TtsPromptCompilerTest, PrependsBosWhenProvided) {
  EXPECT_EQ(compiler::compilePromptString("hello", std::nullopt, {},
                                          "<|begin_of_text|>"),
            "<|begin_of_text|><|bot|>hello" + kReadout + "<|speech_start|>");
}

TEST(TtsPromptCompilerTest, OmitsBosWhenEmpty) {
  EXPECT_EQ(compiler::compilePromptString("hello", std::nullopt, {}, ""),
            "<|bot|>hello" + kReadout + "<|speech_start|>");
}

// BOS leads the whole prompt — ahead of the audio/voice prompt blocks, not
// just ahead of <|bot|>. The reference compiler emits [BOS, ...] first.
TEST(TtsPromptCompilerTest, BosPrecedesAudioAndVoicePromptBlocks) {
  const std::vector<uint32_t> speechIds = {12, 34};
  EXPECT_EQ(compiler::compilePromptString("hello", std::string("calm voice"),
                                          speechIds, "<|begin_of_text|>"),
            "<|begin_of_text|><|audio_prompt_start|><|s_12|><|s_34|>"
            "<|audio_prompt_end|><|voice_prompt_start|>calm voice"
            "<|voice_prompt_end|><|bot|>hello" +
                kReadout + "<|speech_start|>");
}

// ---- TtsRequest::fromJson: speech_ids ----

Json::Value parseJson(const std::string& text) {
  Json::Value value;
  Json::CharReaderBuilder builder;
  std::string errors;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  EXPECT_TRUE(
      reader->parse(text.data(), text.data() + text.size(), &value, &errors))
      << errors;
  return value;
}

TEST(TtsRequestJsonTest, ParsesSpeechIds) {
  const auto request = tt::domain::tts::TtsRequest::fromJson(
      parseJson(R"({"text": "hi", "speech_ids": [5, 6, 65535]})"), 1);
  EXPECT_EQ(request.promptSpeechIds, (std::vector<uint32_t>{5, 6, 65535}));
  EXPECT_FALSE(request.voiceSample.has_value());
}

TEST(TtsRequestJsonTest, SpeechIdsDefaultsToEmpty) {
  const auto request = tt::domain::tts::TtsRequest::fromJson(
      parseJson(R"({"text": "hi"})"), 1);
  EXPECT_TRUE(request.promptSpeechIds.empty());
}

TEST(TtsRequestJsonTest, RejectsNonArraySpeechIds) {
  EXPECT_THROW(tt::domain::tts::TtsRequest::fromJson(
                   parseJson(R"({"text": "hi", "speech_ids": "5,6"})"), 1),
               std::invalid_argument);
}

TEST(TtsRequestJsonTest, RejectsEmptySpeechIds) {
  EXPECT_THROW(tt::domain::tts::TtsRequest::fromJson(
                   parseJson(R"({"text": "hi", "speech_ids": []})"), 1),
               std::invalid_argument);
}

TEST(TtsRequestJsonTest, RejectsNegativeOrNonIntegerSpeechIds) {
  EXPECT_THROW(tt::domain::tts::TtsRequest::fromJson(
                   parseJson(R"({"text": "hi", "speech_ids": [1, -2]})"), 1),
               std::invalid_argument);
  EXPECT_THROW(tt::domain::tts::TtsRequest::fromJson(
                   parseJson(R"({"text": "hi", "speech_ids": [1.5]})"), 1),
               std::invalid_argument);
}

}  // namespace
