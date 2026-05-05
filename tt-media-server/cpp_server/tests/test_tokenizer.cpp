// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "config/settings.hpp"
#include "domain/chat_message.hpp"
#include "utils/tokenizers/tokenizer.hpp"

using namespace tt::utils::tokenizers;
using namespace tt::domain;

// ---------------------------------------------------------------------------
// Fixture: creates a tokenizer for the env-selected model (encode/decode tests)
// ---------------------------------------------------------------------------

class TokenizerTest : public ::testing::Test {
 protected:
  std::unique_ptr<Tokenizer> tok;

  void SetUp() override {
    std::string path = tt::config::tokenizerPath();
    if (path.empty()) {
      FAIL() << "Tokenizer not found for model: "
             << activeTokenizer().modelName();
    }
    tok = createTokenizer(tt::config::modelType(), path);
    if (!tok->isLoaded()) {
      FAIL() << "Failed to load tokenizer from: " << path;
    }
  }

  Tokenizer& tokenizer() { return *tok; }
};

TEST_F(TokenizerTest, EncodeDecodeRoundTrip) {
  std::string prompt = "The quick brown fox jumps over the lazy dog.";
  auto tokens = tokenizer().encode(prompt);
  EXPECT_GT(tokens.size(), 0);
  std::string decoded = tokenizer().decode(tokens);
  EXPECT_FALSE(decoded.empty());
}

TEST_F(TokenizerTest, EmptyStringEncode) {
  auto tokens = tokenizer().encode("");
  EXPECT_EQ(tokens.size(), 0);
}

TEST_F(TokenizerTest, EmptyTokensDecode) {
  std::string decoded = tokenizer().decode({});
  EXPECT_EQ(decoded, "");
}

class DeepseekTokenizerTest : public ::testing::Test {
 protected:
  std::unique_ptr<Tokenizer> tok;

  void SetUp() override {
    std::string path =
        tt::config::tokenizerPath(tt::config::ModelType::DEEPSEEK_R1_0528);
    if (path.empty()) {
      GTEST_SKIP() << "DeepSeek tokenizer files not found";
    }
    tok = createTokenizer(tt::config::ModelType::DEEPSEEK_R1_0528, path);
    if (!tok->isLoaded()) {
      FAIL() << "Failed to load DeepSeek tokenizer from: " << path;
    }
  }

  Tokenizer& tokenizer() { return *tok; }
};

TEST_F(DeepseekTokenizerTest, CompareWithExpectedTokens) {
  // Pre-computed expected tokens from DeepSeek R1 05 28 tokenizer (without
  // special tokens) Generated using: tokenizer.encode(text,
  // add_special_tokens=False)
  std::map<std::string, std::vector<int>> expectedTokens = {
      {"Hello, world!", {19923, 14, 2058, 3}},
      {"The quick brown fox jumps over the lazy dog.",
       {671, 4787, 13769, 46012, 54994, 1060, 270, 41638, 6397, 16}},
      {"Machine learning and artificial intelligence are transforming "
       "industries.",
       {56891, 3607, 305, 16500, 12967, 477, 38892, 15668, 16}},
      {"In a world where technology advances rapidly, innovation is key.",
       {1124, 260, 2058, 1479, 4807, 25038, 14647, 14, 13194, 344, 3077, 16}},
      {"Python is a high-level programming language.",
       {36914, 344, 260, 1669, 12675, 14051, 4063, 16}},
      {"Deep learning models require large amounts of data.",
       {53091, 3607, 5363, 3506, 3226, 13469, 294, 1499, 16}},
      {"Natural language processing enables computers to understand human "
       "language.",
       {30852, 4063, 8037, 17689, 19578, 304, 2572, 2883, 4063, 16}},
      {"Tokenization is the first step in text processing.",
       {14907, 1878, 344, 270, 1257, 3132, 295, 3051, 8037, 16}},
      {"Transformers have revolutionized the field of NLP.",
       {46998, 387, 611, 76146, 270, 2994, 294, 93526, 16}},
      {"Attention mechanisms allow models to focus on relevant information.",
       {108558, 12187, 2534, 5363, 304, 3568, 377, 7723, 1951, 16}},
      {"The model was trained on a diverse dataset.",
       {671, 2645, 515, 17024, 377, 260, 10445, 20071, 16}},
      {"Inference speed is crucial for production deployments.",
       {1124, 2838, 6276, 344, 7648, 362, 4606, 110998, 16}},
      {"Optimization techniques improve model performance.",
       {80655, 1878, 7189, 5831, 2645, 4197, 16}},
      {"Gradient descent is used to minimize the loss function.",
       {41, 53343, 38655, 344, 1505, 304, 21896, 270, 4721, 2019, 16}},
      {"Neural networks consist of interconnected layers of neurons.",
       {11067, 1614, 11024, 5184, 294, 45639, 14177, 294, 22833, 16}},
      {"Backpropagation computes gradients for weight updates.",
       {12939, 26702, 64148, 89946, 59773, 362, 5288, 17745, 16}},
      {"Overfitting occurs when a model memorizes training data.",
       {7853, 112984, 10122, 1082, 260, 2645, 19607, 6530, 5026, 1499, 16}},
      {"Regularization helps prevent overfitting.",
       {51549, 1878, 7531, 4819, 1060, 112984, 16}},
      {"Dropout randomly disables neurons during training.",
       {50152, 606, 28467, 787, 3208, 22833, 2184, 5026, 16}},
      {"Batch normalization stabilizes training.",
       {83469, 67908, 18745, 6530, 5026, 16}},
      {"Learning rate schedules adjust the step size over time.",
       {36518, 3711, 38318, 7486, 270, 3132, 3701, 1060, 1014, 16}},
      {"Early stopping prevents unnecessary training iterations.",
       {41337, 30308, 30685, 28148, 5026, 53678, 16}},
      {"Cross-validation assesses model generalization.",
       {10138, 116561, 86961, 2645, 59859, 16}},
      {"Ensemble methods combine multiple models for better predictions.",
       {60939, 15300, 4836, 20036, 4990, 5363, 362, 2993, 26145, 16}},
      {"Transfer learning leverages pre-trained models.",
       {53861, 3607, 122686, 852, 76287, 5363, 16}},
      {"Fine-tuning adapts models to specific tasks.",
       {70017, 108073, 6708, 85, 5363, 304, 3549, 10017, 16}},
      {"Zero-shot learning performs tasks without training examples.",
       {51140, 87520, 3607, 29266, 10017, 2503, 5026, 7165, 16}},
      {"Few-shot learning requires minimal training data.",
       {98073, 87520, 3607, 7391, 17515, 5026, 1499, 16}},
      {"Meta-learning enables learning to learn.",
       {52252, 42854, 17689, 3607, 304, 3281, 16}},
      {"Reinforcement learning optimizes actions through rewards.",
       {2167, 261, 13437, 3607, 7944, 6530, 8102, 1407, 31929, 16}},
      {"Q-learning estimates action values.",
       {51, 42854, 16152, 4271, 3785, 16}},
      {"Policy gradients directly optimize the policy.",
       {41592, 59773, 6578, 27474, 270, 5242, 16}},
      {"Actor-critic methods combine value and policy approaches.",
       {96521, 2846, 46178, 4836, 20036, 1990, 305, 5242, 10576, 16}},
      {"Monte Carlo methods sample trajectories.",
       {14956, 592, 42707, 4836, 6810, 57498, 16}},
      {"Temporal difference learning updates estimates incrementally.",
       {54, 46119, 5335, 3607, 17745, 16152, 35133, 1101, 16}},
      {"Exploration balances trying new actions with exploiting known ones.",
       {13978, 8702, 47668, 5958, 1017, 8102, 418, 74888, 3459, 6684, 16}},
      {"Exploitation uses the best-known strategy.",
       {13978, 86688, 6623, 270, 2455, 20814, 7822, 16}},
      {"Multi-armed bandits model the exploration-exploitation tradeoff.",
       {37460, 15, 52121, 6762, 1303, 2645, 270, 18355, 120427, 86688, 7629,
        4676, 16}},
      {"Contextual bandits consider additional state information.",
       {9914, 814, 6762, 1303, 2255, 5974, 2501, 1951, 16}},
      {"Markov decision processes formalize sequential decision problems.",
       {17567, 757, 5227, 6579, 10956, 1387, 44479, 5227, 4454, 16}},
      {"State spaces represent all possible configurations.",
       {6878, 13564, 3293, 710, 3338, 35826, 16}},
      {"Action spaces define available choices.",
       {15656, 13564, 11348, 3510, 13239, 16}},
      {"Reward functions encode objectives.",
       {63560, 593, 6177, 57395, 15417, 16}},
      {"Discount factors weight future rewards.",
       {88789, 3687, 5288, 3988, 31929, 16}},
      {"Value functions estimate expected returns.",
       {6685, 6177, 13236, 5604, 10340, 16}},
      {"Bellman equations express recursive value relationships.",
       {45182, 2160, 11702, 4651, 50494, 1990, 8561, 16}},
      {"Dynamic programming solves MDPs exactly.",
       {72836, 14051, 83029, 373, 9422, 85, 9045, 16}},
      {"Model-free methods learn without environment models.",
       {8449, 13697, 4836, 3281, 2503, 3431, 5363, 16}},
      {"Model-based methods leverage environment dynamics.",
       {8449, 4890, 4836, 30150, 3431, 14520, 16}},
      {"Sample efficiency measures data requirements.",
       {40433, 9062, 7809, 1499, 7172, 16}},
  };

  size_t totalMatches = 0;
  size_t totalMismatches = 0;

  for (const auto& [prompt, expected] : expectedTokens) {
    auto cppTokens = tokenizer().encode(prompt);

    EXPECT_GT(cppTokens.size(), 0) << "Empty C++ tokens for: " << prompt;

    if (cppTokens == expected) {
      ++totalMatches;
    } else {
      ++totalMismatches;

      std::stringstream cppSs, expSs;
      cppSs << "[";
      for (size_t j = 0; j < cppTokens.size(); ++j) {
        cppSs << cppTokens[j];
        if (j < cppTokens.size() - 1) cppSs << ", ";
      }
      cppSs << "]";

      expSs << "[";
      for (size_t j = 0; j < expected.size(); ++j) {
        expSs << expected[j];
        if (j < expected.size() - 1) expSs << ", ";
      }
      expSs << "]";

      EXPECT_EQ(cppTokens, expected)
          << "Token mismatch for prompt: \"" << prompt << "\"\n"
          << "  C++ tokens:      " << cppSs.str() << "\n"
          << "  Expected tokens: " << expSs.str();
    }
  }

  std::cout << "\nTokenizer Validation Summary:\n";
  std::cout << "  Total prompts:  " << expectedTokens.size() << "\n";
  std::cout << "  Matches:        " << totalMatches << "\n";
  std::cout << "  Mismatches:     " << totalMismatches << "\n";
}

TEST_F(DeepseekTokenizerTest, ApplyChatTemplateMatchesDeepSeekR10528Format) {
  // Same message list as used in HuggingFace docs for apply_chat_template.
  std::vector<ChatMessage> messages = {
      {"user", "Hello"},
      {"assistant", "Hi!"},
      {"user", "How are you?"},
  };

  // Expected output from HuggingFace transformers
  // tokenizer.apply_chat_template(..., add_generation_prompt=True) for
  // DeepSeek-R1-0528 (add_bos_token=true, add_eos_token=false).
  const std::string expected =
      "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>Hi!<｜User｜>How "
      "are you?<｜Assistant｜>";

  std::string actual = tokenizer().applyChatTemplate(messages, true);

  EXPECT_EQ(actual, expected)

      << "apply_chat_template output should match HuggingFace DeepSeek-R1-0528 "
         "format.\n"
      << "  Expected length: " << expected.size() << "\n"
      << "  Actual length:   " << actual.size();
}

TEST_F(DeepseekTokenizerTest,
       ApplyChatTemplateReasoningDisabledInjectsThinkBlock) {
  std::vector<ChatMessage> messages = {
      {"user", "Hello"},
  };

  const std::string expected =
      "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>"
      "<think>\n</think>\n";

  std::string actual =
      tokenizer().applyChatTemplate(messages, true, std::nullopt, false);

  EXPECT_EQ(actual, expected)
      << "enable_reasoning=false should inject a closed <think> block after "
         "the assistant tag.\n"
      << "  Expected length: " << expected.size() << "\n"
      << "  Actual length:   " << actual.size();
}

TEST_F(DeepseekTokenizerTest, ApplyChatTemplateReasoningEnabledNoThinkBlock) {
  std::vector<ChatMessage> messages = {
      {"user", "Hello"},
  };

  std::string actual =
      tokenizer().applyChatTemplate(messages, true, std::nullopt, true);

  EXPECT_EQ(actual.find("<think>"), std::string::npos)
      << "enable_reasoning=true should not inject a <think> block";
}

TEST_F(DeepseekTokenizerTest,
       ApplyChatTemplateNoGenerationPromptMatchesDeepSeekR10528Format) {
  // Same message list as used in HuggingFace docs for apply_chat_template.
  std::vector<ChatMessage> messages = {
      {"user", "Hello"},
      {"assistant", "Hi!"},
      {"user", "How are you?"},
  };

  // Expected output from HuggingFace transformers
  // tokenizer.apply_chat_template(..., add_generation_prompt=True) for
  // DeepSeek-R1-0528 (add_bos_token=true, add_eos_token=false).
  const std::string expected =
      "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>Hi!<｜User｜>How "
      "are you?";

  std::string actual = tokenizer().applyChatTemplate(messages, false);

  EXPECT_EQ(actual, expected)

      << "apply_chat_template output should match HuggingFace DeepSeek-R1-0528 "
         "format.\n"
      << "  Expected length: " << expected.size() << "\n"
      << "  Actual length:   " << actual.size();
}

// ---------------------------------------------------------------------------
// StreamDecoder tests
// ---------------------------------------------------------------------------

TEST_F(DeepseekTokenizerTest, StreamDecoderEmoji) {
  // 🛑 encodes to 3 tokens [3574, 252, 242] in DeepSeek
  auto ids = tokenizer().encode("\xF0\x9F\x9B\x91");  // 🛑
  ASSERT_GE(ids.size(), 2) << "Emoji should be multiple tokens";

  auto decoder = tokenizer().createStreamDecoder();
  std::string accumulated;
  for (size_t i = 0; i < ids.size(); ++i) {
    std::string delta = decoder->step(ids[i]);
    accumulated += delta;
    if (i < ids.size() - 1) {
      EXPECT_EQ(delta, "")
          << "Intermediate byte-fragment tokens should be buffered";
    }
  }
  accumulated += decoder->flush();
  EXPECT_EQ(accumulated, "\xF0\x9F\x9B\x91");  // 🛑
}

TEST_F(DeepseekTokenizerTest, StreamDecoderAscii) {
  std::string text = "Hello world";
  auto ids = tokenizer().encode(text);

  auto decoder = tokenizer().createStreamDecoder();
  std::string accumulated;
  for (int id : ids) {
    accumulated += decoder->step(id);
  }
  accumulated += decoder->flush();
  EXPECT_EQ(accumulated, text);
}

TEST_F(DeepseekTokenizerTest, StreamDecoderMixedMultibyte) {
  // Mix of ASCII + CJK + emoji
  std::string text = "Hi\xe4\xbd\xa0\xe5\xa5\xbd\xF0\x9F\x9B\x91";  // Hi你好🛑
  auto ids = tokenizer().encode(text);

  auto decoder = tokenizer().createStreamDecoder();
  std::string accumulated;
  for (int id : ids) {
    accumulated += decoder->step(id);
  }
  accumulated += decoder->flush();
  EXPECT_EQ(accumulated, text);
}

TEST_F(DeepseekTokenizerTest, StreamDecoderFlushIncomplete) {
  // Simulate a single byte-fragment token that never completes.
  // flush() should still return something (possibly with U+FFFD).
  auto ids = tokenizer().encode("\xF0\x9F\x9B\x91");  // 🛑
  ASSERT_GE(ids.size(), 2);

  auto decoder = tokenizer().createStreamDecoder();
  std::string delta = decoder->step(ids[0]);
  EXPECT_EQ(delta, "");

  std::string flushed = decoder->flush();
  EXPECT_FALSE(flushed.empty()) << "flush() should emit buffered content";
}

TEST_F(DeepseekTokenizerTest, StreamDecoderMatchesBatchDecode) {
  // For any token sequence, streaming decode should produce the same
  // final text as a single batch decode.
  std::string text =
      "Roger\xe6\x89\x80\xe6\xb1\x82 fieldwork\xe6\xa0\xb8\xe8\x8b\xb7";
  auto ids = tokenizer().encode(text);

  auto decoder = tokenizer().createStreamDecoder();
  std::string streamed;
  for (int id : ids) {
    streamed += decoder->step(id);
  }
  streamed += decoder->flush();

  std::string batched = tokenizer().decode(ids);
  EXPECT_EQ(streamed, batched);
}

// ---------------------------------------------------------------------------
// skip_special_tokens tests
// ---------------------------------------------------------------------------

TEST_F(DeepseekTokenizerTest, DecodeSpecialTokenSkipped) {
  // Token ID 1 is <｜end▁of▁sentence｜> — a special token in DeepSeek
  std::string skipped = tokenizer().decode({1}, /*skip_special_tokens=*/true);
  EXPECT_EQ(skipped, "");
}

TEST_F(DeepseekTokenizerTest, DecodeSpecialTokenPreserved) {
  std::string kept = tokenizer().decode({1}, /*skip_special_tokens=*/false);
  EXPECT_FALSE(kept.empty())
      << "Decoding a special token with skip=false should produce text";
}

TEST_F(DeepseekTokenizerTest, DecodeSpecialTokenMixedWithNormal) {
  // "Hello" tokens followed by the EOS special token
  auto helloIds = tokenizer().encode("Hello");
  ASSERT_FALSE(helloIds.empty());

  std::vector<int> withSpecial = helloIds;
  withSpecial.push_back(1);  // append EOS

  std::string skipped =
      tokenizer().decode(withSpecial, /*skip_special_tokens=*/true);
  std::string kept =
      tokenizer().decode(withSpecial, /*skip_special_tokens=*/false);

  EXPECT_EQ(skipped, tokenizer().decode(helloIds))
      << "Skipping special tokens should produce same result as without them";
  EXPECT_GT(kept.size(), skipped.size())
      << "Preserving special tokens should produce longer output";
}

TEST_F(DeepseekTokenizerTest, StreamDecoderSkipSpecialTokens) {
  auto helloIds = tokenizer().encode("Hello");
  ASSERT_FALSE(helloIds.empty());
  helloIds.push_back(1);  // append EOS special token

  auto decoder = tokenizer().createStreamDecoder(/*skip_special_tokens=*/true);
  std::string result;
  for (int id : helloIds) result += decoder->step(id);
  result += decoder->flush();

  EXPECT_NE(result.find("Hello"), std::string::npos);
  // The special token text should not appear
  std::string eosText = tokenizer().decode({1}, /*skip_special_tokens=*/false);
  EXPECT_EQ(result.find(eosText), std::string::npos)
      << "Special token text should be absent when skip=true";
}

TEST_F(DeepseekTokenizerTest, StreamDecoderPreserveSpecialTokens) {
  auto helloIds = tokenizer().encode("Hello");
  ASSERT_FALSE(helloIds.empty());
  helloIds.push_back(1);  // append EOS special token

  auto decoder = tokenizer().createStreamDecoder(/*skip_special_tokens=*/false);
  std::string result;
  for (int id : helloIds) result += decoder->step(id);
  result += decoder->flush();

  std::string eosText = tokenizer().decode({1}, /*skip_special_tokens=*/false);
  EXPECT_NE(result.find(eosText), std::string::npos)
      << "Special token text should be present when skip=false";
}

TEST_F(DeepseekTokenizerTest, StreamDecoderMatchesBatchDecodeWithSkipSpecial) {
  auto helloIds = tokenizer().encode("Hello world");
  helloIds.push_back(1);

  for (bool skip : {true, false}) {
    auto decoder = tokenizer().createStreamDecoder(skip);
    std::string streamed;
    for (int id : helloIds) streamed += decoder->step(id);
    streamed += decoder->flush();

    std::string batched = tokenizer().decode(helloIds, skip);
    EXPECT_EQ(streamed, batched)
        << "StreamDecoder should match batch decode (skip=" << skip << ")";
  }
}

// ---------------------------------------------------------------------------
// Fixture: Llama-specific tests (always creates a Llama tokenizer)
// ---------------------------------------------------------------------------

class LlamaTokenizerTest : public ::testing::Test {
 protected:
  std::unique_ptr<Tokenizer> tok;

  void SetUp() override {
    std::string path =
        tt::config::tokenizerPath(tt::config::ModelType::LLAMA_3_1_8B_INSTRUCT);
    if (path.empty()) {
      GTEST_SKIP() << "Llama tokenizer files not found";
    }
    tok = createTokenizer(tt::config::ModelType::LLAMA_3_1_8B_INSTRUCT, path);
    if (!tok->isLoaded()) {
      FAIL() << "Failed to load Llama tokenizer from: " << path;
    }
  }

  Tokenizer& tokenizer() { return *tok; }
};

TEST_F(LlamaTokenizerTest, ApplyChatTemplate) {
  std::vector<ChatMessage> messages = {
      {"user", "Hello"},
      {"assistant", "Hi!"},
      {"user", "How are you?"},
  };

  const std::string expected =
      "<|begin_of_text|>"
      "<|start_header_id|>system<|end_header_id|>\n\n"
      "Cutting Knowledge Date: December 2023\n"
      "Today Date: 26 Jul 2024\n\n"
      "<|eot_id|>"
      "<|start_header_id|>user<|end_header_id|>\n\n"
      "Hello<|eot_id|>"
      "<|start_header_id|>assistant<|end_header_id|>\n\n"
      "Hi!<|eot_id|>"
      "<|start_header_id|>user<|end_header_id|>\n\n"
      "How are you?<|eot_id|>"
      "<|start_header_id|>assistant<|end_header_id|>\n\n";

  std::string actual = tokenizer().applyChatTemplate(messages, true);

  EXPECT_EQ(actual, expected)
      << "apply_chat_template output should match Llama 3.1 8B Instruct "
         "format.\n"
      << "  Expected length: " << expected.size() << "\n"
      << "  Actual length:   " << actual.size();
}

TEST_F(LlamaTokenizerTest, ApplyChatTemplateNoGenerationPrompt) {
  std::vector<ChatMessage> messages = {
      {"user", "Hello"},
      {"assistant", "Hi!"},
      {"user", "How are you?"},
  };

  const std::string expected =
      "<|begin_of_text|>"
      "<|start_header_id|>system<|end_header_id|>\n\n"
      "Cutting Knowledge Date: December 2023\n"
      "Today Date: 26 Jul 2024\n\n"
      "<|eot_id|>"
      "<|start_header_id|>user<|end_header_id|>\n\n"
      "Hello<|eot_id|>"
      "<|start_header_id|>assistant<|end_header_id|>\n\n"
      "Hi!<|eot_id|>"
      "<|start_header_id|>user<|end_header_id|>\n\n"
      "How are you?<|eot_id|>";

  std::string actual = tokenizer().applyChatTemplate(messages, false);

  EXPECT_EQ(actual, expected)
      << "apply_chat_template output should match Llama 3.1 8B Instruct "
         "format.\n"
      << "  Expected length: " << expected.size() << "\n"
      << "  Actual length:   " << actual.size();
}
