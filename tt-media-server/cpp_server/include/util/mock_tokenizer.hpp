// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>
#include <vector>

namespace tt::util {

/**
 * Mock tokenizer for prompt tokenization.
 * Character-level encoding: each byte maps to token id 0-255.
 * Reversible via detokenize for testing/development without a real tokenizer.
 */
class MockTokenizer {
public:
    static std::vector<int> tokenize(const std::string& text);
    static std::string detokenize(const std::vector<int>& token_ids);
};

}  // namespace tt::util
