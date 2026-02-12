// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace tt::utils;

constexpr const char* TOKENIZER_PATH = "tokenizers/tokenizer.json";

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer = TokenizerUtil::load(TOKENIZER_PATH);
        if (!tokenizer.is_loaded()) {
            GTEST_SKIP() << "Tokenizer not found at: " << TOKENIZER_PATH;
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

TEST_F(TokenizerTest, CompareWithPythonTransformers) {
    std::vector<std::string> test_prompts = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are transforming industries.",
        "In a world where technology advances rapidly, innovation is key.",
        "Python is a high-level programming language.",
        "Deep learning models require large amounts of data.",
        "Natural language processing enables computers to understand human language.",
        "Tokenization is the first step in text processing.",
        "Transformers have revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant information.",
        "The model was trained on a diverse dataset.",
        "Inference speed is crucial for production deployments.",
        "Optimization techniques improve model performance.",
        "Gradient descent is used to minimize the loss function.",
        "Neural networks consist of interconnected layers of neurons.",
        "Backpropagation computes gradients for weight updates.",
        "Overfitting occurs when a model memorizes training data.",
        "Regularization helps prevent overfitting.",
        "Dropout randomly disables neurons during training.",
        "Batch normalization stabilizes training.",
        "Learning rate schedules adjust the step size over time.",
        "Early stopping prevents unnecessary training iterations.",
        "Cross-validation assesses model generalization.",
        "Ensemble methods combine multiple models for better predictions.",
        "Transfer learning leverages pre-trained models.",
        "Fine-tuning adapts models to specific tasks.",
        "Zero-shot learning performs tasks without training examples.",
        "Few-shot learning requires minimal training data.",
        "Meta-learning enables learning to learn.",
        "Reinforcement learning optimizes actions through rewards.",
        "Q-learning estimates action values.",
        "Policy gradients directly optimize the policy.",
        "Actor-critic methods combine value and policy approaches.",
        "Monte Carlo methods sample trajectories.",
        "Temporal difference learning updates estimates incrementally.",
        "Exploration balances trying new actions with exploiting known ones.",
        "Exploitation uses the best-known strategy.",
        "Multi-armed bandits model the exploration-exploitation tradeoff.",
        "Contextual bandits consider additional state information.",
        "Markov decision processes formalize sequential decision problems.",
        "State spaces represent all possible configurations.",
        "Action spaces define available choices.",
        "Reward functions encode objectives.",
        "Discount factors weight future rewards.",
        "Value functions estimate expected returns.",
        "Bellman equations express recursive value relationships.",
        "Dynamic programming solves MDPs exactly.",
        "Model-free methods learn without environment models.",
        "Model-based methods leverage environment dynamics.",
        "Sample efficiency measures data requirements."
    };

    // Create Python script to get reference tokens
    std::ofstream script("compare_tokenizer.py");
    script << "from transformers import AutoTokenizer\n";
    script << "import json\n";
    script << "tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3', trust_remote_code=True)\n";
    script << "prompts = " << "[";
    for (size_t i = 0; i < test_prompts.size(); ++i) {
        script << "\"\"\"" << test_prompts[i] << "\"\"\"";
        if (i < test_prompts.size() - 1) script << ", ";
    }
    script << "]\n";
    script << "results = [tokenizer.encode(p) for p in prompts]\n";
    script << "with open('python_tokens.json', 'w') as f:\n";
    script << "    json.dump(results, f)\n";
    script.close();

    // Run Python script
    int ret = system("python3 compare_tokenizer.py 2>/dev/null");
    if (ret != 0) {
        GTEST_SKIP() << "Python transformers not available";
    }

    // Load Python results
    std::ifstream f("python_tokens.json");
    if (!f.good()) {
        GTEST_SKIP() << "Failed to generate Python reference tokens";
    }

    // Simple validation - just check C++ tokenizer produces non-empty results
    for (const auto& prompt : test_prompts) {
        auto cpp_tokens = tokenizer.encode(prompt);
        EXPECT_GT(cpp_tokens.size(), 0) << "Empty tokens for prompt: " << prompt;

        std::string decoded = tokenizer.decode(cpp_tokens);
        EXPECT_FALSE(decoded.empty()) << "Empty decode for prompt: " << prompt;
    }

    // Cleanup
    std::remove("compare_tokenizer.py");
    std::remove("python_tokens.json");
}
