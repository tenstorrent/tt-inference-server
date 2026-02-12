// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "config/settings.hpp"

#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>

using namespace tt::utils;

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::string tokenizer_file_path = tt::config::tokenizer_path();
        if (tokenizer_file_path.empty()) {
            GTEST_SKIP() << "Tokenizer not found at default location (tokenizers/tokenizer.json)";
        }
        tokenizer = TokenizerUtil::load(tokenizer_file_path);
        if (!tokenizer.is_loaded()) {
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
    script << "results = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]\n";
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

    rapidjson::IStreamWrapper isw(f);
    rapidjson::Document doc;
    doc.ParseStream(isw);
    f.close();

    if (!doc.IsArray()) {
        GTEST_SKIP() << "Invalid Python tokens JSON format";
    }

    ASSERT_EQ(doc.Size(), test_prompts.size()) << "Mismatch in number of results";

    // Compare C++ tokenizer with Python transformers
    size_t total_matches = 0;
    size_t total_mismatches = 0;

    for (size_t i = 0; i < test_prompts.size(); ++i) {
        const auto& prompt = test_prompts[i];
        auto cpp_tokens = tokenizer.encode(prompt);

        ASSERT_TRUE(doc[i].IsArray()) << "Invalid token array at index " << i;
        const auto& py_array = doc[i].GetArray();

        std::vector<int> py_tokens;
        for (const auto& token : py_array) {
            ASSERT_TRUE(token.IsInt()) << "Non-integer token in Python result";
            py_tokens.push_back(token.GetInt());
        }

        EXPECT_GT(cpp_tokens.size(), 0) << "Empty C++ tokens for: " << prompt;
        EXPECT_GT(py_tokens.size(), 0) << "Empty Python tokens for: " << prompt;

        if (cpp_tokens == py_tokens) {
            ++total_matches;
        } else {
            ++total_mismatches;

            std::stringstream cpp_ss, py_ss;
            cpp_ss << "[";
            for (size_t j = 0; j < cpp_tokens.size(); ++j) {
                cpp_ss << cpp_tokens[j];
                if (j < cpp_tokens.size() - 1) cpp_ss << ", ";
            }
            cpp_ss << "]";

            py_ss << "[";
            for (size_t j = 0; j < py_tokens.size(); ++j) {
                py_ss << py_tokens[j];
                if (j < py_tokens.size() - 1) py_ss << ", ";
            }
            py_ss << "]";

            EXPECT_EQ(cpp_tokens, py_tokens)
                << "Token mismatch for prompt: \"" << prompt << "\"\n"
                << "  C++ tokens:    " << cpp_ss.str() << "\n"
                << "  Python tokens: " << py_ss.str();
        }
    }

    // Summary
    std::cout << "\nTokenizer Comparison Summary:\n";
    std::cout << "  Total prompts:  " << test_prompts.size() << "\n";
    std::cout << "  Matches:        " << total_matches << "\n";
    std::cout << "  Mismatches:     " << total_mismatches << "\n";

    // Cleanup
    std::remove("compare_tokenizer.py");
    std::remove("python_tokens.json");
}
