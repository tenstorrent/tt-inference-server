// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tts_prompt_compiler.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "utils/tokenizers/tokenizer.hpp"

namespace {

namespace compiler = tt::utils::tts_prompt_compiler;

TEST(TtsPromptCompilerTest, CompilesTextOnlyPrompt) {
  EXPECT_EQ(compiler::compilePromptString("  hello there  ", std::nullopt),
            "<|bot|>hello there<|speech_start|>");
}

TEST(TtsPromptCompilerTest, CompilesDescriptionPrompt) {
  EXPECT_EQ(
      compiler::compilePromptString("hello", std::string("  calm voice ")),
      "<|voice_prompt_start|>calm voice<|voice_prompt_end|>"
      "<|bot|>hello<|speech_start|>");
}

TEST(TtsPromptCompilerTest, CompilesVoiceSamplePrompt) {
  const std::vector<uint32_t> speechIds = {12, 34, 56};
  EXPECT_EQ(compiler::compilePromptString("hello", std::string("calm voice"),
                                          speechIds),
            "<|audio_prompt_start|><|s_12|><|s_34|><|s_56|>"
            "<|audio_prompt_end|><|voice_prompt_start|>calm voice"
            "<|voice_prompt_end|><|bot|>hello<|speech_start|>");
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
            "<|begin_of_text|><|bot|>hello<|speech_start|>");
}

TEST(TtsPromptCompilerTest, OmitsBosWhenEmpty) {
  EXPECT_EQ(compiler::compilePromptString("hello", std::nullopt, {}, ""),
            "<|bot|>hello<|speech_start|>");
}

TEST(TtsPromptCompilerTest, BosPrecedesAudioAndVoicePromptBlocks) {
  const std::vector<uint32_t> speechIds = {12, 34};
  EXPECT_EQ(compiler::compilePromptString("hello", std::string("calm voice"),
                                          speechIds, "<|begin_of_text|>"),
            "<|begin_of_text|><|audio_prompt_start|><|s_12|><|s_34|>"
            "<|audio_prompt_end|><|voice_prompt_start|>calm voice"
            "<|voice_prompt_end|><|bot|>hello<|speech_start|>");
}

}  // namespace
