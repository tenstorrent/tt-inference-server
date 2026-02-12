// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "config/settings.hpp"

#include <gtest/gtest.h>
#include <map>
#include <vector>
#include <string>
#include <sstream>

using namespace tt::utils;

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::string tokenizer_file_path = tt::config::tokenizer_path();
        if (tokenizer_file_path.empty()) {
            GTEST_SKIP() << "Tokenizer not found at default location (tokenizers/tokenizer.json)";
        }
        tokenizer = TokenizerUtil(tokenizer_file_path);

        // Check if tokenizer loaded successfully by testing encode
        if (tokenizer.encode("test").empty()) {
            GTEST_SKIP() << "Failed to load tokenizer from: " << tokenizer_file_path;
        }
    }

    TokenizerUtil tokenizer;
};

TEST_F(TokenizerTest, EncodeDecodeRoundTrip) {
    std::string prompt = "The quick brown fox jumps over the lazy dog.";

    auto tokens = tokenizer.encode(prompt);
    EXPECT_GT(tokens.size(), 0);

    std::string decoded = tokenizer.decode(tokens);
    EXPECT_FALSE(decoded.empty());
}

TEST_F(TokenizerTest, EmptyStringEncode) {
    auto tokens = tokenizer.encode("");
    EXPECT_EQ(tokens.size(), 0);
}

TEST_F(TokenizerTest, EmptyTokensDecode) {
    std::string decoded = tokenizer.decode({});
    EXPECT_EQ(decoded, "");
}

TEST_F(TokenizerTest, CompareWithExpectedTokens) {
    // Pre-computed expected tokens from DeepSeek V3 tokenizer (without special tokens)
    // Generated using: tokenizer.encode(text, add_special_tokens=False)
    std::map<std::string, std::vector<int>> expected_tokens = {
        {"Hello, world!", {19923, 14, 2058, 3}},
        {"The quick brown fox jumps over the lazy dog.", {671, 4787, 13769, 46012, 54994, 1060, 270, 41638, 6397, 16}},
        {"Machine learning and artificial intelligence are transforming industries.", {56891, 3607, 305, 16500, 12967, 477, 38892, 15668, 16}},
        {"In a world where technology advances rapidly, innovation is key.", {1124, 260, 2058, 1479, 4807, 25038, 14647, 14, 13194, 344, 3077, 16}},
        {"Python is a high-level programming language.", {36914, 344, 260, 1669, 12675, 14051, 4063, 16}},
        {"Deep learning models require large amounts of data.", {53091, 3607, 5363, 3506, 3226, 13469, 294, 1499, 16}},
        {"Natural language processing enables computers to understand human language.", {30852, 4063, 8037, 17689, 19578, 304, 2572, 2883, 4063, 16}},
        {"Tokenization is the first step in text processing.", {14907, 1878, 344, 270, 1257, 3132, 295, 3051, 8037, 16}},
        {"Transformers have revolutionized the field of NLP.", {46998, 387, 611, 76146, 270, 2994, 294, 93526, 16}},
        {"Attention mechanisms allow models to focus on relevant information.", {108558, 12187, 2534, 5363, 304, 3568, 377, 7723, 1951, 16}},
        {"The model was trained on a diverse dataset.", {671, 2645, 515, 17024, 377, 260, 10445, 20071, 16}},
        {"Inference speed is crucial for production deployments.", {1124, 2838, 6276, 344, 7648, 362, 4606, 110998, 16}},
        {"Optimization techniques improve model performance.", {80655, 1878, 7189, 5831, 2645, 4197, 16}},
        {"Gradient descent is used to minimize the loss function.", {41, 53343, 38655, 344, 1505, 304, 21896, 270, 4721, 2019, 16}},
        {"Neural networks consist of interconnected layers of neurons.", {11067, 1614, 11024, 5184, 294, 45639, 14177, 294, 22833, 16}},
        {"Backpropagation computes gradients for weight updates.", {12939, 26702, 64148, 89946, 59773, 362, 5288, 17745, 16}},
        {"Overfitting occurs when a model memorizes training data.", {7853, 112984, 10122, 1082, 260, 2645, 19607, 6530, 5026, 1499, 16}},
        {"Regularization helps prevent overfitting.", {51549, 1878, 7531, 4819, 1060, 112984, 16}},
        {"Dropout randomly disables neurons during training.", {50152, 606, 28467, 787, 3208, 22833, 2184, 5026, 16}},
        {"Batch normalization stabilizes training.", {83469, 67908, 18745, 6530, 5026, 16}},
        {"Learning rate schedules adjust the step size over time.", {36518, 3711, 38318, 7486, 270, 3132, 3701, 1060, 1014, 16}},
        {"Early stopping prevents unnecessary training iterations.", {41337, 30308, 30685, 28148, 5026, 53678, 16}},
        {"Cross-validation assesses model generalization.", {10138, 116561, 86961, 2645, 59859, 16}},
        {"Ensemble methods combine multiple models for better predictions.", {60939, 15300, 4836, 20036, 4990, 5363, 362, 2993, 26145, 16}},
        {"Transfer learning leverages pre-trained models.", {53861, 3607, 122686, 852, 76287, 5363, 16}},
        {"Fine-tuning adapts models to specific tasks.", {70017, 108073, 6708, 85, 5363, 304, 3549, 10017, 16}},
        {"Zero-shot learning performs tasks without training examples.", {51140, 87520, 3607, 29266, 10017, 2503, 5026, 7165, 16}},
        {"Few-shot learning requires minimal training data.", {98073, 87520, 3607, 7391, 17515, 5026, 1499, 16}},
        {"Meta-learning enables learning to learn.", {52252, 42854, 17689, 3607, 304, 3281, 16}},
        {"Reinforcement learning optimizes actions through rewards.", {2167, 261, 13437, 3607, 7944, 6530, 8102, 1407, 31929, 16}},
        {"Q-learning estimates action values.", {51, 42854, 16152, 4271, 3785, 16}},
        {"Policy gradients directly optimize the policy.", {41592, 59773, 6578, 27474, 270, 5242, 16}},
        {"Actor-critic methods combine value and policy approaches.", {96521, 2846, 46178, 4836, 20036, 1990, 305, 5242, 10576, 16}},
        {"Monte Carlo methods sample trajectories.", {14956, 592, 42707, 4836, 6810, 57498, 16}},
        {"Temporal difference learning updates estimates incrementally.", {54, 46119, 5335, 3607, 17745, 16152, 35133, 1101, 16}},
        {"Exploration balances trying new actions with exploiting known ones.", {13978, 8702, 47668, 5958, 1017, 8102, 418, 74888, 3459, 6684, 16}},
        {"Exploitation uses the best-known strategy.", {13978, 86688, 6623, 270, 2455, 20814, 7822, 16}},
        {"Multi-armed bandits model the exploration-exploitation tradeoff.", {37460, 15, 52121, 6762, 1303, 2645, 270, 18355, 120427, 86688, 7629, 4676, 16}},
        {"Contextual bandits consider additional state information.", {9914, 814, 6762, 1303, 2255, 5974, 2501, 1951, 16}},
        {"Markov decision processes formalize sequential decision problems.", {17567, 757, 5227, 6579, 10956, 1387, 44479, 5227, 4454, 16}},
        {"State spaces represent all possible configurations.", {6878, 13564, 3293, 710, 3338, 35826, 16}},
        {"Action spaces define available choices.", {15656, 13564, 11348, 3510, 13239, 16}},
        {"Reward functions encode objectives.", {63560, 593, 6177, 57395, 15417, 16}},
        {"Discount factors weight future rewards.", {88789, 3687, 5288, 3988, 31929, 16}},
        {"Value functions estimate expected returns.", {6685, 6177, 13236, 5604, 10340, 16}},
        {"Bellman equations express recursive value relationships.", {45182, 2160, 11702, 4651, 50494, 1990, 8561, 16}},
        {"Dynamic programming solves MDPs exactly.", {72836, 14051, 83029, 373, 9422, 85, 9045, 16}},
        {"Model-free methods learn without environment models.", {8449, 13697, 4836, 3281, 2503, 3431, 5363, 16}},
        {"Model-based methods leverage environment dynamics.", {8449, 4890, 4836, 30150, 3431, 14520, 16}},
        {"Sample efficiency measures data requirements.", {40433, 9062, 7809, 1499, 7172, 16}},
    };

    size_t total_matches = 0;
    size_t total_mismatches = 0;

    for (const auto& [prompt, expected] : expected_tokens) {
        auto cpp_tokens = tokenizer.encode(prompt);

        EXPECT_GT(cpp_tokens.size(), 0) << "Empty C++ tokens for: " << prompt;

        if (cpp_tokens == expected) {
            ++total_matches;
        } else {
            ++total_mismatches;

            std::stringstream cpp_ss, exp_ss;
            cpp_ss << "[";
            for (size_t j = 0; j < cpp_tokens.size(); ++j) {
                cpp_ss << cpp_tokens[j];
                if (j < cpp_tokens.size() - 1) cpp_ss << ", ";
            }
            cpp_ss << "]";

            exp_ss << "[";
            for (size_t j = 0; j < expected.size(); ++j) {
                exp_ss << expected[j];
                if (j < expected.size() - 1) exp_ss << ", ";
            }
            exp_ss << "]";

            EXPECT_EQ(cpp_tokens, expected)
                << "Token mismatch for prompt: \"" << prompt << "\"\n"
                << "  C++ tokens:      " << cpp_ss.str() << "\n"
                << "  Expected tokens: " << exp_ss.str();
        }
    }

    // Summary
    std::cout << "\nTokenizer Validation Summary:\n";
    std::cout << "  Total prompts:  " << expected_tokens.size() << "\n";
    std::cout << "  Matches:        " << total_matches << "\n";
    std::cout << "  Mismatches:     " << total_mismatches << "\n";
}
